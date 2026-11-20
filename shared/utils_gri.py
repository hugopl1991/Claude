"""
Modulo de funcoes da GRI (Global Reporting Initiative) para gerar as tabelas 106ai e 107ai, bem como a tabela de balanço natural por ReIBGE.
Baseado no script matlab original, mas reescrito em Python para maior flexibilidade e integração com pandas.
"""

import pandas as pd
import numpy as np


def _read_optional_sheet(workbook, sheet_name, columns):
    if sheet_name not in workbook.sheet_names:
        return pd.DataFrame(columns=columns)
    return pd.read_excel(workbook, sheet_name=sheet_name)


def carregar_dados(arquivo_fuca, arquivo_reclass, base_year, end_year, size_pixel, config=None):
    """Lê e prepara as tabelas de comparação e reclassificação da área."""
    config = config or {}
    FUCA = pd.read_excel(arquivo_fuca, sheet_name=0)
    col_lulc_base = f'LULC_{base_year}'
    col_lulc_end = f'LULC_{end_year}'

    with pd.ExcelFile(arquivo_reclass) as xls:
        general_candidates = config.get(
            'lulc_reclass_sheets',
            ['Reclass_LULC_geral', 'Reclass_LULC_canga', 'Reclass_LULC_floresta'],
        )
        Reclass_IBGE = _read_optional_sheet(xls, 'Reclass_IBGE', ['IBGE', 'Reclass_IBGE'])
        Reclass_geral = pd.DataFrame(columns=['LULC', 'Reclass_LULC'])
        for sheet in general_candidates:
            if sheet in xls.sheet_names:
                candidate = pd.read_excel(xls, sheet_name=sheet)
                if {'LULC', 'Reclass_LULC'}.issubset(candidate.columns):
                    Reclass_geral = candidate
                    break
        Reclass_canga = _read_optional_sheet(
            xls, config.get('lulc_canga_sheet', 'Reclass_LULC_canga'),
            ['LULC', 'Reclass_LULC'],
        )
        Reclass_N4N5 = _read_optional_sheet(
            xls, config.get('lulc_n4n5_sheet', 'Reclass_LULC_N4N5'),
            ['LULC', 'Reclass_LULC'],
        )

    required_columns = {col_lulc_base, col_lulc_end, 'Area m2', 'Qtd_pixels', 'OpUnit', 'IBGE'}
    missing_columns = required_columns.difference(FUCA.columns)
    if missing_columns:
        raise ValueError(
            f'Tabela de comparação sem colunas necessárias: {", ".join(missing_columns)}'
        )

    for column in (col_lulc_base, col_lulc_end):
        FUCA[column] = FUCA[column].astype(str).str.strip()
        FUCA.loc[FUCA[column] == "0", column] = 'Zero'

    corrections = config.get(
        'lulc_class_corrections',
        [{'when_end': 'S11D - PARNA', 'set_base': 'S11D'}],
    )

    for correction in corrections:
        idx_correction = FUCA[col_lulc_end] == correction['when_end']
        FUCA.loc[idx_correction, col_lulc_base] = correction['set_base']

    try:
        pixel_size = float(size_pixel)
    except (TypeError, ValueError) as error:
        raise ValueError('size_pixel deve ser um número positivo.') from error
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError('size_pixel deve ser um número positivo.')

    FUCA['Area m2'] = pd.to_numeric(FUCA['Area m2'], errors='raise')
    FUCA['Qtd_pixels'] = pd.to_numeric(FUCA['Qtd_pixels'], errors='raise')
    idx_zero = FUCA['Area m2'] == 0
    FUCA.loc[idx_zero, 'Area m2'] = (
        FUCA.loc[idx_zero, 'Qtd_pixels'] * pixel_size ** 2
    )
    FUCA['Area_ha'] = FUCA['Area m2'] / 10000

    return FUCA, Reclass_IBGE, Reclass_geral, Reclass_canga, Reclass_N4N5


def _lookup_dict(lookup, keycol, valcol):
    if keycol not in lookup.columns or valcol not in lookup.columns:
        raise ValueError(
            f'Tabela de reclassificação precisa conter as colunas '
            f'{keycol} e {valcol}.'
        )
    keys = lookup[keycol].astype(str).str.strip()
    return dict(zip(keys, lookup[valcol]))


def reclass(var, opunit, ibge, lookup_geral, lookup_N4N5, lookup_canga, keycol, valcol, config=None):
    """Aplica as regras de reclassificação definidas para a área."""
    config = config or {}
    var_txt = var.astype(str).str.strip()
    opunit_txt = opunit.astype(str).str.strip().str.upper()
    ibge_txt = ibge.astype(str).str.strip().str.casefold()

    dict_geral = _lookup_dict(lookup_geral, keycol, valcol)
    dict_N4N5 = _lookup_dict(lookup_N4N5, keycol, valcol)
    dict_canga = _lookup_dict(lookup_canga, keycol, valcol)

    canga_ibge_classes = {
        str(value).strip().casefold()
        for value in config.get(
            'canga_ibge_classes', ['Refúgio Vegetacional Montano']
        )
    }
    is_refugio = ibge_txt.isin(canga_ibge_classes)
    opunits_com_canga = opunit_txt[is_refugio].unique()

    n4n5_operational_units = {
        str(value).strip().upper()
        for value in config.get('n4n5_operational_units', ['N4N5'])
    }
    is_N4N5 = opunit_txt.isin(n4n5_operational_units)
    is_canga_opunit = opunit_txt.isin(opunits_com_canga)

    out = var_txt.map(dict_geral)
    idx_N4N5 = is_N4N5
    idx_canga = ~is_N4N5 & is_canga_opunit
    for mask, lookup in ((idx_N4N5, dict_N4N5), (idx_canga, dict_canga)):
        specialized = var_txt[mask].map(lookup)
        out.loc[mask] = specialized.combine_first(out.loc[mask])

    return out


def gerar_tabelas_intermediarias(FUCA, path_out_106, path_out_107, path_out_reclass, base_year, end_year, sep, decimal, float_format):
    """Gera e exporta as tabelas 106ai e 107ai usando colunas dinâmicas para os anos."""
    col_relulc_base = f'ReLULC_{base_year}'
    col_relulc_end = f'ReLULC_{end_year}'
    col_lulc_end = f'LULC_{end_year}'
    col_cond_base = f'Condicao_{base_year}'
    col_cond_end = f'Condicao_{end_year}'
    col_recond_base = f'ReCond_{base_year}'
    col_recond_end = f'ReCond_{end_year}'

    ReLULCBase_txt = FUCA[col_relulc_base].astype(str).str.strip().str.lower()
    ReLULCEnd_txt = FUCA[col_relulc_end].astype(str).str.strip().str.lower()
    
    # ==========================================
    # Tabela 106ai
    # ==========================================
    nat_changed = (ReLULCBase_txt != "antrópico") & (ReLULCBase_txt != ReLULCEnd_txt)
    FUCA_nat_changed = FUCA[nat_changed]
    
    G106 = FUCA_nat_changed.groupby(['OpUnit', 'ReIBGE', col_relulc_base, col_relulc_end], as_index=False).agg(
        GroupCount=('OpUnit', 'size'),
        sum_Area_ha=('Area_ha', 'sum')
    )

    G106b = FUCA_nat_changed.groupby(['OpUnit', 'ReIBGE', col_relulc_base, col_lulc_end], as_index=False).agg(
        GroupCount=('OpUnit', 'size'),
        sum_Area_ha=('Area_ha', 'sum')
    )

    # ==========================================
    # Tabela 107ai
    # ==========================================
    antBase = ReLULCBase_txt == "antrópico"
    antEnd = ReLULCEnd_txt == "antrópico"
    
    FUCA[col_recond_base] = np.where(antBase, 'Zero', FUCA[col_cond_base])
    FUCA[col_recond_end] = np.where(antEnd, 'Zero', FUCA[col_cond_end])

    G107b = FUCA.groupby(['OpUnit', 'ReIBGE', col_recond_base, col_recond_end], as_index=False).agg(
        GroupCount=('OpUnit', 'size'),
        sum_Area_ha=('Area_ha', 'sum')
    )

    # Exportando com os parâmetros configurados no yaml
    G106.to_csv(path_out_106, index=False, float_format=float_format, sep=sep, decimal=decimal)
    G107b.to_csv(path_out_107, index=False, float_format=float_format, sep=sep, decimal=decimal)
    FUCA.to_csv(path_out_reclass, index=False, float_format=float_format, sep=sep, decimal=decimal)
   
    return antBase, antEnd, G106b


def calcular_balanco(FUCA, antBase, antEnd, base_year, end_year):
    """Calcula o balanço natural, antrópico e total por OpUnit usando anos dinâmicos."""
    col_area_base = f'Area{base_year}'
    col_area_end = f'Area{end_year}'
    
    isNatBase, isNatEnd = ~antBase, ~antEnd

    # 1. Classes Naturais
    baseNat = FUCA[['OpUnit', 'ReIBGE']].drop_duplicates()
    G_Base = FUCA[isNatBase].groupby(['OpUnit', 'ReIBGE'], as_index=False)['Area_ha'].sum().rename(columns={'Area_ha': col_area_base})
    G_End = FUCA[isNatEnd].groupby(['OpUnit', 'ReIBGE'], as_index=False)['Area_ha'].sum().rename(columns={'Area_ha': col_area_end})
    Gganho = FUCA[~isNatBase & isNatEnd].groupby(['OpUnit', 'ReIBGE'], as_index=False)['Area_ha'].sum().rename(columns={'Area_ha': 'Ganho'})
    Gperda = FUCA[isNatBase & ~isNatEnd].groupby(['OpUnit', 'ReIBGE'], as_index=False)['Area_ha'].sum().rename(columns={'Area_ha': 'Perda'})

    BalancoNat = baseNat.merge(G_Base, on=['OpUnit', 'ReIBGE'], how='left')\
                        .merge(Gganho, on=['OpUnit', 'ReIBGE'], how='left')\
                        .merge(Gperda, on=['OpUnit', 'ReIBGE'], how='left')\
                        .merge(G_End, on=['OpUnit', 'ReIBGE'], how='left').fillna(0)

    BalancoNat['Ganho_pct'] = np.where(BalancoNat[col_area_base] > 0, BalancoNat['Ganho'] / BalancoNat[col_area_base], 0)
    BalancoNat['Perda_pct'] = np.where(BalancoNat[col_area_base] > 0, BalancoNat['Perda'] / BalancoNat[col_area_base], 0)

    idxNaturais = (~BalancoNat['ReIBGE'].str.strip().str.lower().isin(["antrópico", "total"])) & (BalancoNat['ReIBGE'].str.strip() != "")
    BalancoNat = BalancoNat[idxNaturais].copy()

    # 2. Antrópico
    baseOp = FUCA[['OpUnit']].drop_duplicates()
    GAntBase = FUCA[~isNatBase].groupby('OpUnit', as_index=False)['Area_ha'].sum().rename(columns={'Area_ha': col_area_base})
    GAntEnd = FUCA[~isNatEnd].groupby('OpUnit', as_index=False)['Area_ha'].sum().rename(columns={'Area_ha': col_area_end})
    
    BalancoAnt = baseOp.merge(GAntBase, on='OpUnit', how='left').merge(GAntEnd, on='OpUnit', how='left').fillna(0)
    BalancoAnt = BalancoAnt.assign(ReIBGE="Antrópico", Ganho=0.0, Ganho_pct=np.nan, Perda=0.0, Perda_pct=np.nan)

    # 3. Total
    BalancoTot = FUCA.groupby('OpUnit', as_index=False)['Area_ha'].sum().rename(columns={'Area_ha': col_area_base})
    BalancoTot[col_area_end] = BalancoTot[col_area_base]
    
    GGanhoTot = BalancoNat.groupby('OpUnit', as_index=False)['Ganho'].sum()
    GPerdaTot = BalancoNat.groupby('OpUnit', as_index=False)['Perda'].sum()
    
    BalancoTot = BalancoTot.merge(GGanhoTot, on='OpUnit', how='left').merge(GPerdaTot, on='OpUnit', how='left').fillna(0)
    BalancoTot['Ganho_pct'] = np.where(BalancoTot[col_area_base] > 0, BalancoTot['Ganho'] / BalancoTot[col_area_base], 0)
    BalancoTot['Perda_pct'] = np.where(BalancoTot[col_area_base] > 0, BalancoTot['Perda'] / BalancoTot[col_area_base], 0)
    BalancoTot['ReIBGE'] = "Total"

    # 4. Formatação Final
    col_out_base = f'Area_{base_year}'
    col_out_end = f'Area_{end_year}'

    rename_cols = {'ReIBGE': 'Classe', col_area_base: col_out_base, col_area_end: col_out_end}
    frames = [df.rename(columns=rename_cols) for df in [BalancoNat, BalancoAnt, BalancoTot]]
    
    cols_order = ['OpUnit', 'Classe', col_out_base, 'Ganho', 'Ganho_pct', 'Perda', 'Perda_pct', col_out_end]
    SaidaBalanco = pd.concat(frames, ignore_index=True)[cols_order]

    ordem = {"Antrópico": 2, "Total": 3}
    SaidaBalanco['ordClasse'] = SaidaBalanco['Classe'].map(ordem).fillna(1)
    
    return SaidaBalanco.sort_values(by=['OpUnit', 'ordClasse', 'Classe']).drop(columns=['ordClasse'])


def _validate_rad_columns(data, col_rad_base, col_rad_end):
    """Valida e converte os indicadores RAD; retorna False se ambos estiverem ausentes."""
    present = (col_rad_base in data.columns, col_rad_end in data.columns)
    if not any(present):
        return False
    if not all(present):
        raise ValueError(
            f'Tabela de comparação precisa conter ambas as colunas RAD: '
            f'{col_rad_base} e {col_rad_end}.'
        )

    for column in (col_rad_base, col_rad_end):
        data[column] = pd.to_numeric(data[column], errors='raise')
        if (
            data[column].isna().any()
            or not np.isfinite(data[column]).all()
            or not data[column].isin([0, 1]).all()
        ):
            raise ValueError(f'A coluna {column} deve conter apenas valores 0 ou 1.')
    return True


def calcular_balanco_bv(FUCA, antBase, antEnd, base_year, end_year, config=None):
    """Calcula balanços de BV natural e RAD, incluindo as variações entre anos."""
    col_bv_base = f'Soma_BV_base_{base_year}'
    col_bv_end = f'Soma_BV_end_{end_year}'
    col_rad_base = f'RAD_{base_year}'
    col_rad_end = f'RAD_{end_year}'
    missing_columns = [
        col for col in (col_bv_base, col_bv_end) if col not in FUCA.columns
    ]
    if missing_columns:
        raise ValueError(
            f'Tabela de comparação sem colunas BV necessárias: {", ".join(missing_columns)}'
        )

    data = FUCA.copy()
    for column in (col_bv_base, col_bv_end):
        data[column] = pd.to_numeric(data[column], errors='raise')
        if data[column].isna().any() or not np.isfinite(data[column]).all():
            raise ValueError(
                f'A coluna {column} contém valores ausentes ou não finitos.'
            )

    has_rad = _validate_rad_columns(data, col_rad_base, col_rad_end)
    if has_rad:
        rad_base = data[col_rad_base] == 1
        rad_end = data[col_rad_end] == 1
    else:
        rad_base = pd.Series(False, index=data.index)
        rad_end = pd.Series(False, index=data.index)

    config = config or {}
    water_opunits = config.get('bv_water_natural_opunits', [])
    if isinstance(water_opunits, str) or not isinstance(
        water_opunits, (list, tuple, set)
    ):
        raise ValueError('bv_water_natural_opunits deve ser uma lista de OpUnits.')

    is_nat_base, is_nat_end = ~antBase, ~antEnd
    if water_opunits:
        col_lulc_base = f'LULC_{base_year}'
        col_lulc_end = f'LULC_{end_year}'
        missing_lulc_columns = [
            col for col in (col_lulc_base, col_lulc_end) if col not in data.columns
        ]
        if missing_lulc_columns:
            raise ValueError(
                f'Tabela de comparação sem colunas LULC necessárias: '
                f'{", ".join(missing_lulc_columns)}'
            )

        selected_opunits = {
            str(opunit).strip().upper() for opunit in water_opunits
        }
        selected_rows = (
            data['OpUnit'].astype(str).str.strip().str.upper().isin(selected_opunits)
        )
        water_class_names = {'33', '33.0', 'rio, lago e oceano'}

        def is_water_class(column):
            lulc_text = data[column].astype(str).str.strip().str.casefold()
            return lulc_text.isin(water_class_names) | lulc_text.str.match(
                r'^33(?:\.0)?\s*[-–:]\s*rio,\s*lago e oceano$'
            )

        water_base = selected_rows & is_water_class(col_lulc_base)
        water_end = selected_rows & is_water_class(col_lulc_end)
        is_nat_base = is_nat_base | water_base
        is_nat_end = is_nat_end | water_end

    is_nat_base &= ~rad_base
    is_nat_end &= ~rad_end

    bv_base_name = f'BV_{base_year}'
    bv_end_name = f'BV_{end_year}'

    base_natural = data[['OpUnit', 'ReIBGE']].drop_duplicates()
    bv_base = (
        data[is_nat_base]
        .groupby(['OpUnit', 'ReIBGE'], as_index=False)[col_bv_base]
        .sum()
        .rename(columns={col_bv_base: bv_base_name})
    )
    bv_end = (
        data[is_nat_end]
        .groupby(['OpUnit', 'ReIBGE'], as_index=False)[col_bv_end]
        .sum()
        .rename(columns={col_bv_end: bv_end_name})
    )
    natural_both_years = is_nat_base & is_nat_end
    bv_change = data[col_bv_end] - data[col_bv_base]
    gained_natural = ~is_nat_base & is_nat_end
    lost_natural = is_nat_base & ~is_nat_end
    data['Ganho'] = 0.0
    data.loc[gained_natural, 'Ganho'] = data.loc[gained_natural, col_bv_end]
    data.loc[natural_both_years, 'Ganho'] = bv_change.clip(lower=0)
    data['Perda'] = 0.0
    data.loc[lost_natural, 'Perda'] = data.loc[lost_natural, col_bv_base]
    data.loc[natural_both_years, 'Perda'] = (-bv_change).clip(lower=0)
    gain_loss_by_class = (
        data.groupby(['OpUnit', 'ReIBGE'], as_index=False)[['Ganho', 'Perda']]
        .sum()
    )

    balance_natural = (
        base_natural.merge(bv_base, on=['OpUnit', 'ReIBGE'], how='left')
        .merge(gain_loss_by_class, on=['OpUnit', 'ReIBGE'], how='left')
        .merge(bv_end, on=['OpUnit', 'ReIBGE'], how='left')
        .fillna(0)
    )
    balance_natural['Ganho_pct'] = np.where(
        balance_natural[bv_base_name] > 0,
        balance_natural['Ganho'] / balance_natural[bv_base_name],
        0,
    )
    balance_natural['Perda_pct'] = np.where(
        balance_natural[bv_base_name] > 0,
        balance_natural['Perda'] / balance_natural[bv_base_name],
        0,
    )
    ibge_text = balance_natural['ReIBGE'].astype(str).str.strip().str.lower()
    natural_classes = ~ibge_text.isin(['antrópico', 'total']) & ibge_text.ne('')
    balance_natural = balance_natural[natural_classes].copy()

    base_opunits = data[['OpUnit']].drop_duplicates()
    bv_anthropic_base = (
        data[~is_nat_base & ~rad_base]
        .groupby('OpUnit', as_index=False)[col_bv_base]
        .sum()
        .rename(columns={col_bv_base: bv_base_name})
    )
    bv_anthropic_end = (
        data[~is_nat_end & ~rad_end]
        .groupby('OpUnit', as_index=False)[col_bv_end]
        .sum()
        .rename(columns={col_bv_end: bv_end_name})
    )
    balance_anthropic = (
        base_opunits.merge(bv_anthropic_base, on='OpUnit', how='left')
        .merge(bv_anthropic_end, on='OpUnit', how='left')
        .fillna(0)
    )
    balance_anthropic = balance_anthropic.assign(
        ReIBGE='Antrópico',
        Ganho=0.0,
        Ganho_pct=np.nan,
        Perda=0.0,
        Perda_pct=np.nan,
    )

    balance_rad = pd.DataFrame()
    if has_rad:
        active_rad = rad_base | rad_end
        if active_rad.any():
            rad_change = data[col_bv_end] - data[col_bv_base]
            entered_rad = ~rad_base & rad_end
            left_rad = rad_base & ~rad_end
            remained_rad = rad_base & rad_end
            rad_gain = np.where(
                entered_rad,
                data[col_bv_end],
                np.where(remained_rad, rad_change.clip(lower=0), 0),
            )
            rad_loss = np.where(
                left_rad,
                data[col_bv_base],
                np.where(remained_rad, (-rad_change).clip(lower=0), 0),
            )
            rad_opunits = data.loc[active_rad, ['OpUnit']].drop_duplicates()
            rad_base_bv = (
                data[rad_base]
                .groupby('OpUnit', as_index=False)[col_bv_base]
                .sum()
                .rename(columns={col_bv_base: bv_base_name})
            )
            rad_end_bv = (
                data[rad_end]
                .groupby('OpUnit', as_index=False)[col_bv_end]
                .sum()
                .rename(columns={col_bv_end: bv_end_name})
            )
            rad_changes = data[['OpUnit']].copy()
            rad_changes['Ganho'] = rad_gain
            rad_changes['Perda'] = rad_loss
            rad_gain_loss = rad_changes.groupby(
                'OpUnit', as_index=False
            )[['Ganho', 'Perda']].sum()
            balance_rad = (
                rad_opunits.merge(rad_base_bv, on='OpUnit', how='left')
                .merge(rad_gain_loss, on='OpUnit', how='left')
                .merge(rad_end_bv, on='OpUnit', how='left')
                .fillna(0)
            )
            balance_rad['Ganho_pct'] = np.where(
                balance_rad[bv_end_name] > 0,
                balance_rad['Ganho'] / balance_rad[bv_end_name],
                0,
            )
            balance_rad['Perda_pct'] = np.where(
                balance_rad[bv_base_name] > 0,
                balance_rad['Perda'] / balance_rad[bv_base_name],
                0,
            )
            balance_rad['ReIBGE'] = 'RAD'

    balance_total = (
        data.groupby('OpUnit', as_index=False)[[col_bv_base, col_bv_end]]
        .sum()
        .rename(columns={col_bv_base: bv_base_name, col_bv_end: bv_end_name})
    )
    total_gain = balance_natural.groupby('OpUnit', as_index=False)['Ganho'].sum()
    total_loss = balance_natural.groupby('OpUnit', as_index=False)['Perda'].sum()
    balance_total = (
        balance_total.merge(total_gain, on='OpUnit', how='left')
        .merge(total_loss, on='OpUnit', how='left')
        .fillna(0)
    )
    balance_total['Ganho_pct'] = np.where(
        balance_total[bv_base_name] > 0,
        balance_total['Ganho'] / balance_total[bv_base_name],
        0,
    )
    balance_total['Perda_pct'] = np.where(
        balance_total[bv_base_name] > 0,
        balance_total['Perda'] / balance_total[bv_base_name],
        0,
    )
    balance_total['ReIBGE'] = 'Total'

    rename_columns = {'ReIBGE': 'Classe'}
    columns_order = [
        'OpUnit',
        'Classe',
        bv_base_name,
        'Ganho',
        'Ganho_pct',
        'Perda',
        'Perda_pct',
        bv_end_name,
    ]
    frames = [
        frame.rename(columns=rename_columns)
        for frame in (balance_natural, balance_anthropic, balance_rad, balance_total)
    ]
    output = pd.concat(frames, ignore_index=True)[columns_order]
    class_order = {'RAD': 2, 'Antrópico': 3, 'Total': 4}
    output['ordClasse'] = output['Classe'].map(class_order).fillna(1)

    return output.sort_values(
        by=['OpUnit', 'ordClasse', 'Classe']
    ).drop(columns=['ordClasse'])