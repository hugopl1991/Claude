"""Reclassificação de MapBiomas para valor biótico (BV), aplicação da condição e da RAD.

Saída: uint8, valores válidos de 1 a 100, nodata = 0.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from shapely.validation import make_valid


def _round(x):
    """Arredonda metade para cima (0.5 -> 1)."""
    return np.floor(x + 0.5)


def load_biotic_coefficients(path: str) -> dict[int, float]:
    """Lê a tabela MapBiomas -> BV (colunas id_class;bv). Classes sem BV são ignoradas."""
    table_path = Path(path)
    if not table_path.is_file():
        raise FileNotFoundError(f"Tabela de BV não encontrada: {table_path}")

    table = pd.read_csv(table_path, sep=";").rename(columns={"id_class": "mapbiomas_class"})
    if not {"mapbiomas_class", "bv"} <= set(table.columns):
        raise ValueError("Tabela de BV precisa das colunas id_class e bv.")

    classes = pd.to_numeric(table["mapbiomas_class"], errors="coerce")
    if classes.isna().any() or (classes % 1 != 0).any():
        raise ValueError("Tabela de BV contém classe não numérica ou não inteira.")
    if classes.duplicated().any():
        raise ValueError("Tabela de BV contém classes duplicadas.")

    text = table["bv"].astype("string").str.strip().str.casefold()
    missing = text.isna() | text.isin(["", "none"])
    bv = pd.to_numeric(table["bv"].mask(missing), errors="coerce")
    if (bv.isna() & ~missing).any():
        raise ValueError("Tabela de BV contém valor não numérico.")
    if not bv.dropna().between(0, 1).all():
        raise ValueError("Valores de BV devem estar no intervalo 0-1.")

    ok = bv.notna()
    return {int(c): float(v) for c, v in zip(classes[ok], bv[ok])}


def map_biotic_value(
    lulc: np.ndarray,
    coefficients: dict[int, float],
    *,
    lulc_nodata: int | float | None,
    nodata_classes: set[int],
    nodata_bv: int,
    min_value: int,
    max_value: int,
) -> np.ndarray:
    """Reclassifica o MapBiomas para BV (min-max). Pixels sem BV recebem nodata_bv."""
    excluded = set(nodata_classes) | {0}
    if lulc_nodata is not None:
        excluded.add(lulc_nodata)

    bv = np.full(lulc.shape, nodata_bv, dtype=np.uint8)
    for class_id, value in coefficients.items():
        if class_id not in excluded:
            bv[lulc == class_id] = int(_round(min_value + value * (max_value - min_value)))
    return bv


def apply_condition(
    bv: np.ndarray,
    condition: np.ndarray,
    *,
    condition_nodata: int | float | None,
    condition_scale: float,
    nodata_bv: int,
    min_value: int,
    max_value: int,
) -> np.ndarray:
    """BV x condição / escala. Onde a condição é NoData, mantém o BV original."""
    if bv.shape != condition.shape:
        raise ValueError("BV e condição precisam ter a mesma dimensão.")
    if condition_scale <= 0:
        raise ValueError("condition_scale deve ser maior que zero.")

    condition_ok = np.isfinite(condition)
    if condition_nodata is not None:
        condition_ok &= condition != condition_nodata
    use = (bv != nodata_bv) & condition_ok

    result = bv.copy()
    product = bv[use].astype(np.float64) * condition[use] / condition_scale
    result[use] = np.clip(_round(product), min_value, max_value).astype(np.uint8)
    return result


def apply_rad_value(
    result: np.ndarray,
    rad_mask: np.ndarray,
    *,
    rad_value: float,
    nodata_bv: int,
    min_value: int,
    max_value: int,
) -> np.ndarray:
    """Substitui pelo BV fixo da RAD todos os pixels da máscara que têm dado."""
    if result.shape != rad_mask.shape:
        raise ValueError("Resultado e máscara de RAD precisam ter a mesma dimensão.")
    if not min_value <= rad_value <= max_value:
        raise ValueError(f"BV de RAD deve estar no intervalo {min_value}-{max_value}.")

    output = result.copy()
    output[rad_mask & (result != nodata_bv)] = int(_round(rad_value))
    return output


def rasterize_rad_shape(
    shape_path: str,
    reference_path: str,
    coverage_threshold: float = 0.5,
    factor: int = 10,
    tile_size: int = 256,
) -> np.ndarray:
    """Máscara dos pixels com cobertura RAD >= limiar, estimada por subpixels."""
    if not 0 < coverage_threshold <= 1:
        raise ValueError("O limiar de cobertura RAD deve estar no intervalo (0, 1].")

    with rasterio.open(reference_path) as ref:
        crs, transform = ref.crs, ref.transform
        height, width = ref.shape

    shapes = gpd.read_file(shape_path)
    if shapes.crs is None or crs is None:
        raise ValueError("Shape de RAD ou raster de referência sem CRS definido.")
    if shapes.empty:
        raise ValueError(f"Shape de RAD sem feições: {shape_path}")

    geoms = shapes.to_crs(crs).geometry
    geoms = geoms[geoms.notna() & ~geoms.is_empty].apply(make_valid)
    geoms = geoms.explode().reset_index(drop=True)
    geoms = geoms[geoms.geom_type == "Polygon"].reset_index(drop=True)
    if geoms.empty:
        raise ValueError(f"Shape de RAD sem polígonos válidos: {shape_path}")

    sindex = geoms.sindex
    min_coverage = np.ceil(coverage_threshold * factor**2 - 1e-9) / factor**2
    mask = np.zeros((height, width), dtype=bool)

    for row in range(0, height, tile_size):
        th = min(tile_size, height - row)
        for col in range(0, width, tile_size):
            tw = min(tile_size, width - col)
            window = Window(col, row, tw, th)

            idx = list(sindex.intersection(window_bounds(window, transform)))
            if not idx:
                continue

            fine = rasterize(
                ((g, 1) for g in geoms.iloc[idx]),
                out_shape=(th * factor, tw * factor),
                transform=window_transform(window, transform) * Affine.scale(1 / factor),
                fill=0,
                dtype=np.uint8,
            )
            count = fine.reshape(th, factor, tw, factor).sum(axis=(1, 3), dtype=np.uint16)
            coverage = count / factor**2
            mask[row : row + th, col : col + tw] = coverage >= min_coverage

    return mask


def write_raster(
    path: str,
    template: str,
    data: np.ndarray,
    description: str,
    nodata_bv: int,
) -> str:
    """Grava um raster uint8 com o georreferenciamento do template."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(template) as source:
        profile = source.profile.copy()
    profile.update(dtype="uint8", count=1, nodata=nodata_bv, compress="lzw")
    profile.pop("predictor", None)  # pode ser incompatível com o dtype do template

    with rasterio.open(output_path, "w", **profile) as dest:
        dest.write(data.astype(np.uint8, copy=False), 1)
        dest.set_band_description(1, description)
    return str(output_path)