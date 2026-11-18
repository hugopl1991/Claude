from pathlib import Path

import yaml

from shared.functions import clip_raster, create_dir, get_raster_band
from biotic_value import (
    apply_condition,
    apply_rad_value,
    load_biotic_coefficients,
    map_biotic_value,
    rasterize_rad_shape,
    write_raster,
)


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    paths, data = cfg["Paths"], cfg["Data"]

    area, year = data["area"], data["end_year"]
    nodata_bv = data["biotic_value_nodata"]
    vmin, vmax = data["axis_min_bv"], data["axis_max_bv"]

    out_path = create_dir(paths["out_dir"])
    tmp_path = create_dir(paths["tmp_dir"])

    fmt = {"AREA": area, "YEAR": year, "END_YEAR": year, "BASE_YEAR": year}
    shp_file = paths["aio_shp_path"].format(**fmt)

    condition_key = (
        "milhao_map_file_base" if year == data["base_year_compare"] else "milhao_map_file_end"
    )
    condition_file = f"{out_path}/{paths[condition_key].format(**fmt)}"

    # Recorte pelo shape da área operacional
    clip = dict(shp_path=shp_file, tmp_path=tmp_path, all_touched=data["all_touched_cond"])
    lulc_map = clip_raster(raster_path=paths["lulc_path"].format(**fmt), **clip)
    condition_map = clip_raster(raster_path=condition_file, **clip)

    lulc, lulc_nodata = get_raster_band(lulc_map)
    condition, condition_nodata = get_raster_band(condition_map)
    if condition_nodata is None:
        condition_nodata = data["nodata_cond"]
    if lulc.shape != condition.shape:
        raise ValueError("Condição e MapBiomas recortados têm dimensões diferentes.")

    # Valor biótico por classe e valor final (BV x condição)
    coefficients = load_biotic_coefficients(paths["biotic_value_table"].format(**fmt))
    bv = map_biotic_value(
        lulc,
        coefficients,
        lulc_nodata=lulc_nodata,
        nodata_classes=set(data["lulc_nodata_classes"]),
        nodata_bv=nodata_bv,
        min_value=vmin,
        max_value=vmax,
    )
    bv_final = apply_condition(
        bv,
        condition,
        condition_nodata=condition_nodata,
        condition_scale=data["condition_scale"],
        nodata_bv=nodata_bv,
        min_value=vmin,
        max_value=vmax,
    )

    # Áreas de RAD são opcionais; sem geometria ou saída configurada, não são aplicadas.
    rad_shape = paths.get("rad_shape_file")
    rad_mask_name = paths.get("rad_mask_name")
    rad_mask = None
    if rad_shape and rad_mask_name:
        rad_shape_path = Path(rad_shape.format(**fmt))
        if rad_shape_path.is_file():
            rad_mask = rasterize_rad_shape(
                str(rad_shape_path), lulc_map, data["rad_coverage_threshold"]
            )
        else:
            print(f"[AVISO] Arquivo RAD não encontrado; nenhum valor RAD aplicado: {rad_shape_path}")
    elif rad_shape:
        print("[AVISO] rad_mask_name não configurado; valores RAD não serão aplicados.")

    if rad_mask is not None:
        # Pixels onde o valor de RAD foi de fato aplicado (1 = RAD, 0 = nodata)
        rad_applied = rad_mask & (bv_final != nodata_bv)
        bv_final = apply_rad_value(
            bv_final,
            rad_mask,
            rad_value=data["rad_bv_value"],
            nodata_bv=nodata_bv,
            min_value=vmin,
            max_value=vmax,
        )
    else:
        rad_applied = None

    if rad_mask_name and rad_applied is not None:
        write_raster(
            f"{out_path}/{rad_mask_name.format(**fmt)}",
            lulc_map,
            rad_applied,
            "Pixels com valor de RAD aplicado (1 = RAD)",
            nodata_bv,
        )

    write_raster(
        f"{out_path}/{paths['bv_map_file'].format(**fmt)}",
        lulc_map,
        bv,
        "Valor biotico por classe MapBiomas (1-100)",
        nodata_bv,
    )
    write_raster(
        f"{out_path}/{paths['bv_final_map_file'].format(**fmt)}",
        lulc_map,
        bv_final,
        "Valor biotico final (BV x condicao, 1-100)",
        nodata_bv,
    )

    print(f"\n[OK] Mapas de valor biotico gerados para {area}/{year}!")


if __name__ == "__main__":
    main()