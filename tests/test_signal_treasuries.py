"""
Tests de la fuente COMPUESTA: el indice de estres de vendedores forzados (DATs).

Esta fuente no tiene proveedor —el mNAV no lo publica ninguna API— asi que TODO el valor
esta en la capa pura: identificar que activo tiene cada tesoreria, decidir quien entra en la
cohorte y no usar un dato antes del dia en que se publico. Nada de eso toca red, y por eso
se ejercita aqui con hechos que tienen la forma EXACTA del XBRL de la SEC, copiados de
respuestas medidas el 2026-08-13.

Los tests que valen algo son los que fallarian si alguien "simplificara" una decision
tomada a proposito. Cinco, que son los cinco sitios donde esto se rompe sin dar error:

  - la etiqueta de unidad NO identifica el activo: lo VERIFICA (CleanSpark declara 1.719.000
    unidades `Bitcoin` que valen 58,53 $ cada una, y leer la etiqueta le sumaria a la
    cohorte de BTC un tesoro cien veces mayor que el de Strategy);
  - un hecho se ve el dia en que se PUBLICA (`filed`) y no el dia al que se refiere
    (`end`): confundirlos mete cinco semanas de futuro y no da ningun error;
  - el tesoro se marca a MERCADO y no al valor razonable publicado, que esta marcado al
    cierre del trimestre;
  - la cohorte se define con el BALANCE y no con el mNAV, porque definirla con el mNAV
    truncaria justo la cola que se publica;
  - una companıa archivada treinta veces (la ventana de captura) cuenta UNA.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ai_trader.observation.signal_radar import POLARITY, SignalRadarProvider
from ai_trader.shared.clock import HistoricalClock
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters import treasuries as T
from ai_trader.signals.catalog import CATALOG, ENCODING_EVENT, get_source
from ai_trader.signals.events import EVENT_SPECS, SUFFIX_MAG, encoded_names
from ai_trader.signals.source import raw_record

UTC = timezone.utc
KEY = "dat_mnav"

# Precio de BTC el 2026-06-30, MEDIDO: es el que hace que el precio implicito de Riot
# (58.527,07 $) case y el de CleanSpark (58,53 $) no.
BTC_2026_06_30 = 58_527.0


def _record(entity: str, payload, **extra):
    return raw_record(source=KEY, entity=entity, payload=payload, **extra)


def _price(entity: str, series: str, days) -> list:
    return [
        _record(entity, {"symbol": entity, "day": day, "close": close},
                day=day, request={"series": series})
        for day, close in days
    ]


def _facts(
    cik: int,
    ticker: str,
    *,
    units: float,
    fair_value: float,
    assets: float,
    shares: float,
    unit: str = "Bitcoin",
    as_of: str = "2026-06-30",
    filed: str = "2026-08-03",
    sic: str = "6199",
    **extra,
):
    """Un registro de hechos con la forma exacta de `companyconcept`."""
    def fact(value, unit_name):
        return {unit_name: [{"end": as_of, "val": value, "filed": filed, "form": "10-Q"}]}

    payload = {
        "cik": cik,
        "ticker": ticker,
        "name": ticker,
        "sic": sic,
        "facts": {
            "units": fact(units, unit),
            "fair_value": fact(fair_value, "USD"),
            "assets": fact(assets, "USD"),
            # El recuento de acciones se fecha en la PORTADA del filing, que es unos dias
            # posterior al cierre del trimestre. MEDIDO: SharpLink cierra el 30/06 y firma
            # el recuento el 03/08.
            "shares": {"shares": [{"end": filed, "val": shares, "filed": filed, "form": "10-Q"}]},
        },
    }
    payload.update(extra)
    return _record(ticker, payload, request={"series": T.SERIES_FACTS, "cik": cik})


# =====================================================================================
# La identificacion del activo: la parte que se equivoca
# =====================================================================================


class TestIdentifyAsset:
    """Un NOMBRE identifica y el PRECIO verifica. Ese reparto de papeles es la decision de
    todo el modulo, y estos son los casos MEDIDOS que lo obligaron."""

    PRICES = {
        "BTC": {"2025-12-31": 88_000.0, "2026-06-30": BTC_2026_06_30},
        "ETH": {"2025-12-31": 2_600.0, "2026-06-30": 1_889.0},
        "SOL": {"2025-12-31": 124.4, "2026-06-30": 76.16},
        "AAVE": {"2025-12-31": 150.0, "2026-06-30": 65.0},
        "NEAR": {"2025-12-31": 1.9, "2026-06-30": 1.60},
    }

    def test_a_naming_label_is_verified_and_not_believed(self):
        """EL TEST QUE PROTEGE LA COHORTE ENTERA. MEDIDO 2026-08-13: CleanSpark declara
        1.719.000 unidades etiquetadas `Bitcoin` con 100,6 M$ de valor razonable, o sea
        58,53 $ por unidad. Son 1.719 bitcoins con un error de escala de mil en el propio
        filing (el trimestre anterior declaro 1.641). Creyendo la etiqueta, la cohorte de
        BTC sumaria un tesoro cien veces mayor que el de la mayor tesoreria que existe."""
        asset, why = T.identify_asset(
            "Bitcoin", "CLEANSPARK, INC.", [(58.53, "2026-06-30")], self.PRICES
        )
        assert asset is None
        assert "contradice" in why

    def test_a_label_that_does_match_identifies(self):
        asset, why = T.identify_asset(
            "Btcoin", "Fold Holdings, Inc.", [(BTC_2026_06_30, "2026-06-30")], self.PRICES
        )
        assert asset == "BTC"  # la errata del emisor esta en la tabla porque esta en un filing
        assert "etiqueta" in why

    def test_the_registrant_name_identifies_when_the_label_does_not(self):
        """MEDIDO: dos tercios de las etiquetas no nombran nada (`Integer`, `item`, `pure`,
        `token`). La razon social si: `Solana Co` dice de que habla su balance."""
        asset, why = T.identify_asset(
            "item", "Solana Co", [(83.12, "2026-06-30")], self.PRICES
        )
        assert asset == "SOL" and "razon social" in why

    def test_a_price_alone_never_identifies_anything(self):
        """LA REGLA QUE SE RETIRO, Y POR QUE. Identificar por 'el unico activo del universo
        que cuadra' daba DOS falsos positivos de ocho sobre la cohorte medida: TON Strategy
        Co (Toncoin, ~1,60 $) salia NEAR e Hyperion DeFi (HYPE) salia LTC. 'Unico' solo
        significa algo si el conjunto de candidatos esta completo, y con veinticuatro
        activos frente a miles de tokens no puede estarlo."""
        asset, why = T.identify_asset(
            "Integer", "Forward Industries, Inc.", [(208.71, "2026-06-30")], self.PRICES
        )
        assert asset is None
        assert "nombran ningun activo" in why

    def test_naming_an_asset_that_is_not_traded_is_a_rejection_with_a_reason(self):
        """MEDIDO: hay tesorerias de TON, HYPE, Canton y USDC. Que la tabla las incluya es
        lo que convierte "no se sabe" en "se sabe, y no se opera", que es una respuesta
        distinta y mucho mas util."""
        asset, why = T.identify_asset(
            "Integer", "TON Strategy Co", [(1.60, "2026-06-30")], self.PRICES
        )
        assert asset is None
        assert "TON, que no esta en el universo" in why

    def test_the_verification_uses_every_date_and_not_just_the_last(self):
        """Un trimestre puede coincidir por casualidad; cuatro no. Aqui el nombre dice SOL y
        el precio implicito cuadra en junio y NO en diciembre: la companıa se cae."""
        assert T.identify_asset(
            "item", "Solana Co", [(76.0, "2026-06-30")], self.PRICES
        )[0] == "SOL"
        assert T.identify_asset(
            "item", "Solana Co", [(76.0, "2026-06-30"), (300.0, "2025-12-31")], self.PRICES
        )[0] is None

    def test_a_two_letter_ticker_is_not_hunted_inside_a_company_name(self):
        """`OP` y `UNI` aparecen dentro de palabras que no tienen nada que ver. Ver
        MIN_NAME_TOKEN: en la razon social solo se buscan tokens de tres letras o mas."""
        assert T.asset_named_in("Op Holdings Universal", whole_text=False) == (None, False)

    def test_the_label_is_matched_whole_and_not_by_words(self):
        assert T.asset_named_in("Bitcoin", whole_text=True) == ("BTC", False)
        assert T.asset_named_in("cryptoAsset", whole_text=True) == (None, False)

    def test_without_a_quote_that_day_it_is_a_hole_and_not_a_confirmation(self):
        asset, why = T.identify_asset(
            "Bitcoin", "Any Corp", [(58_500.0, "2024-01-01")], self.PRICES
        )
        assert asset is None and "no hay precio" in why

    @pytest.mark.parametrize("implied", [0.0, -1.0, float("nan")])
    def test_a_degenerate_implied_price_is_not_an_asset(self, implied):
        assert T.identify_asset(
            "Bitcoin", "Any Corp", [(implied, "2026-06-30")], self.PRICES
        )[0] is None


# =====================================================================================
# Anti look-ahead: el dia en que el dato EXISTE
# =====================================================================================


class TestVisibility:
    def _treasury(self, facts):
        return T.Treasury(
            cik=1, ticker="AAA", name="AAA", asset="BTC", units=tuple(facts), shares=(),
            treasury_share_of_assets=0.9, implied_unit_price=BTC_2026_06_30, identified_by="test",
        )

    def test_a_fact_is_visible_the_day_it_is_filed_and_not_the_day_it_refers_to(self):
        """EL NUCLEO. Strategy cerro el trimestre el 30/06 y lo publico el 03/08: entre esas
        dos fechas hay cinco semanas en las que el numero no existia para nadie. Filtrar por
        `end` en vez de por `filed` mete esas cinco semanas de futuro en cada backtest, y no
        produce ningun error ni ninguna fila rara."""
        fact = T.Fact(as_of="2026-06-30", filed="2026-08-03", value=846_000.0)
        treasury = self._treasury([fact])
        assert treasury.visible(treasury.units, "2026-07-15") is None
        assert treasury.visible(treasury.units, "2026-08-03") is fact

    def test_the_realized_lag_is_subtracted_and_not_assumed(self):
        assert T.Fact("2026-06-30", "2026-08-03", 1.0).lag_days == 34

    def test_the_latest_filing_wins_and_a_restatement_does_not_resurrect_the_old_one(self):
        """MEDIDO: las unidades de Strategy a 2025-12-31 vuelven a aparecer en el 10-Q
        publicado el 2026-08-03. Las dos lineas se conservan en el archivo (es como se mide
        cuanto revisa un proveedor) y al derivar gana la publicada mas tarde."""
        old = T.Fact("2026-03-31", "2026-05-06", 762_099.0)
        new = T.Fact("2026-06-30", "2026-08-03", 846_000.0)
        restated = T.Fact("2025-12-31", "2026-08-03", 672_500.0)
        treasury = self._treasury([old, restated, new])
        assert treasury.visible(treasury.units, "2026-08-10") is new

    def test_a_company_that_stopped_filing_drops_out_instead_of_freezing(self):
        """Arrastrar el ultimo tesoro conocido de una companıa que dejo de declarar la
        mantendria en la distribucion con una tenencia de hace un ano, marcada a precio de
        hoy. Ver FACT_STALE_DAYS."""
        stale = T.Fact("2024-06-30", "2024-08-03", 1_000.0)
        treasury = self._treasury([stale])
        assert treasury.visible(treasury.units, "2026-08-13") is None


# =====================================================================================
# Quien entra en la cohorte: el balance, nunca el mNAV
# =====================================================================================


class TestCohortMembership:
    PRICES = {"BTC": {"2026-06-30": BTC_2026_06_30}}

    def test_a_miner_with_some_crypto_is_not_a_treasury(self):
        """MEDIDO 2026-08-13: CleanSpark tiene 100,6 M$ de cripto sobre 2.702 M$ de activo
        (3,7%) y Riot un 22%. Su mNAV es el cociente entre un negocio operativo y una
        partida del balance, no la magnitud de la que habla esta fuente; sin el filtro
        entran con mNAV de dos cifras y desplazan la mediana del grupo entero."""
        records = [_facts(1, "CLSK", units=1_719, fair_value=100_607_000,
                          assets=2_702_000_000, shares=256_817_073)]
        kept, dropped = T.treasuries_from_records(records, self.PRICES)
        assert not kept
        assert "no es una tesoreria" in dropped[0]["reason"]

    def test_the_filter_reads_the_balance_sheet_and_not_the_share_price(self):
        """No es circular a proposito: la misma companıa entra en la cohorte tanto si su
        accion vale tres dolares como si vale trescientos."""
        records = [_facts(1, "AAA", units=1_000, fair_value=1_000 * BTC_2026_06_30,
                          assets=1_000 * BTC_2026_06_30 / 0.9, shares=1_000_000)]
        kept, _ = T.treasuries_from_records(records, self.PRICES)
        assert [t.ticker for t in kept] == ["AAA"]
        assert kept[0].asset == "BTC"

    def test_a_multiclass_filer_is_declared_missing_and_not_patched(self):
        """MEDIDO: el `companyfacts` de Strategy tiene UN tag `dei` (EntityPublicFloat) y
        ningun recuento de acciones ordinarias, porque las APIs XBRL solo exponen hechos sin
        dimensiones. Se declara el hueco. Sustituirlo por la media ponderada del periodo
        —que si esta— sesgaria a la baja la capitalizacion justo de las que mas emiten."""
        record = _facts(1, "MSTR", units=846_000, fair_value=846_000 * BTC_2026_06_30,
                        assets=52_562_592_000, shares=1)
        record["payload"]["facts"]["shares"] = {}
        kept, dropped = T.treasuries_from_records([record], self.PRICES)
        assert not kept
        assert "multiclase" in dropped[0]["reason"]

    def test_the_same_company_archived_thirty_times_is_one_company(self):
        """La captura pide una ventana de treinta dias y re-archiva los mismos hechos cada
        dia. Sin deduplicar por CIK, una tesoreria seria treinta y la fraccion bajo 1 se
        calcularia sobre una cohorte inventada."""
        copies = [
            _facts(1, "AAA", units=1_000, fair_value=1_000 * BTC_2026_06_30,
                   assets=1_000 * BTC_2026_06_30, shares=1_000_000,
                   fetched_at=datetime(2026, 8, 1 + i, tzinfo=UTC))
            for i in range(30)
        ]
        kept, _ = T.treasuries_from_records(copies, self.PRICES)
        assert len(kept) == 1

    def test_a_fair_value_from_another_date_does_not_become_a_unit_price(self):
        """El cociente valor/unidades solo es un precio si las dos patas son del mismo
        cierre contable. Con fechas distintas seria un numero con la forma correcta y sin
        significado, que es justo como se cuela un activo mal identificado."""
        record = _facts(1, "AAA", units=1_000, fair_value=1.0,
                        assets=1_000 * BTC_2026_06_30, shares=1_000_000)
        record["payload"]["facts"]["fair_value"] = {
            "USD": [{"end": "2026-03-31", "val": 1_000 * BTC_2026_06_30,
                     "filed": "2026-05-06", "form": "10-Q"}]
        }
        _, dropped = T.treasuries_from_records([record], self.PRICES)
        assert "sin valor razonable" in dropped[0]["reason"]


# =====================================================================================
# La distribucion: lo que de verdad se publica
# =====================================================================================


def _cohort_records(prices_by_ticker, *, asset="BTC", units=1_000, day="2026-08-03"):
    """Una cohorte completa: hechos de cada companıa mas las dos patas de precio."""
    records = _price(asset, T.SERIES_ASSET_PX,
                     [("2026-06-30", BTC_2026_06_30), (day, BTC_2026_06_30)])
    for i, (ticker, shares, price) in enumerate(prices_by_ticker):
        records.append(
            _facts(100 + i, ticker, units=units, fair_value=units * BTC_2026_06_30,
                   assets=units * BTC_2026_06_30, shares=shares, filed=day)
        )
        records += _price(ticker, T.SERIES_EQUITY_PX, [("2026-06-30", price), (day, price)])
    return records


class TestDistribution:
    def test_the_published_number_is_the_fraction_below_one(self):
        # Tesoro identico (1.000 BTC) y tres capitalizaciones: 0,5x, 1,5x y 2,0x.
        treasury = 1_000 * BTC_2026_06_30
        records = _cohort_records([
            ("AAA", 1_000_000, treasury * 0.5 / 1_000_000),
            ("BBB", 1_000_000, treasury * 1.5 / 1_000_000),
            ("CCC", 1_000_000, treasury * 2.0 / 1_000_000),
        ])
        row = T.cohort_rows(records)[0]
        assert row["dat_below_nav_share"] == pytest.approx(1 / 3)
        assert row["dat_mnav_gap"] == pytest.approx(0.5)  # mediana 1,5x
        assert row["dat_companies"] == 3.0

    def test_the_treasury_is_marked_to_market_and_not_to_the_published_fair_value(self):
        """Las dos patas estan obsoletas de forma distinta: las unidades lo estan en
        CANTIDAD y el valor razonable en cantidad Y en precio. Si la fila usara el valor
        razonable, la cola inferior se moveria con la fecha del filing y no con el mercado:
        aqui BTC cae a la mitad y el mNAV de todos tiene que doblarse."""
        treasury = 1_000 * BTC_2026_06_30
        cohort = [("AAA", 1_000_000, treasury * 0.5 / 1_000_000),
                  ("BBB", 1_000_000, treasury * 0.6 / 1_000_000),
                  ("CCC", 1_000_000, treasury * 0.7 / 1_000_000)]
        crashed = [
            r for r in _cohort_records(cohort)
            if not (r["entity"] == "BTC" and r["payload"]["day"] == "2026-08-03")
        ] + _price("BTC", T.SERIES_ASSET_PX, [("2026-08-03", BTC_2026_06_30 / 2)])
        row = T.cohort_rows(crashed)[0]
        assert row["dat_below_nav_share"] == 0.0  # 1,0x / 1,2x / 1,4x: ninguno bajo 1
        assert row["dat_mnav_gap"] == pytest.approx(0.2)

    def test_the_row_is_dated_at_the_filing_and_not_at_the_period_end(self):
        row = T.cohort_rows(_cohort_records([("AAA", 1_000_000, 1.0), ("BBB", 1_000_000, 2.0),
                                             ("CCC", 1_000_000, 3.0)]))[0]
        assert row[DAY] == "2026-08-03"

    def test_the_published_lag_is_the_realized_one(self):
        row = T.cohort_rows(_cohort_records([("AAA", 1_000_000, 1.0), ("BBB", 1_000_000, 2.0),
                                             ("CCC", 1_000_000, 3.0)]))[0]
        assert row["dat_disclosure_lag_days"] == 34.0  # 2026-06-30 -> 2026-08-03

    def test_two_companies_are_not_a_distribution(self):
        """Con una cohorte de dos, "la fraccion por debajo de 1" solo puede valer 0, 0,5 o 1
        y la palabra distribucion sobra. Ver MIN_COHORT."""
        assert T.cohort_rows(_cohort_records([("AAA", 1_000_000, 1.0),
                                              ("BBB", 1_000_000, 2.0)])) == []

    def test_between_filing_seasons_there_is_no_row(self):
        """La razon por la que la cadencia es de EVENTO: la tenencia solo cambia cuando
        alguien la publica, asi que fuera de la temporada de resultados no hay observacion.
        Una serie continua rellenaria esos dias con el ultimo valor y una z sobre eso
        contaria como informacion lo que es un carry."""
        rows = T.cohort_rows(_cohort_records([("AAA", 1_000_000, 1.0), ("BBB", 1_000_000, 2.0),
                                              ("CCC", 1_000_000, 3.0)]))
        assert len(rows) == 1

    def test_the_frame_keeps_the_declared_column_order(self):
        frame = T.DigitalAssetTreasuries(get_source(KEY)).daily_from_raw(
            _cohort_records([("AAA", 1_000_000, 1.0), ("BBB", 1_000_000, 2.0),
                             ("CCC", 1_000_000, 3.0)])
        )
        assert list(frame.columns)[:5] == list(get_source(KEY).feature_names)
        assert frame.index.get_level_values(ENTITY)[0] == "BTC"


# =====================================================================================
# El informe: la cifra que hay que publicar
# =====================================================================================


class TestCohortReport:
    def test_the_pooled_n_is_company_observations_and_not_companies(self):
        """La muestra de esta fuente no la dan los eventos de una companıa —cada una publica
        cuatro veces al ano— sino el POOLING sobre la cohorte. Las dos cifras van juntas
        porque doscientas observaciones de tres companias y doscientas de cuarenta sostienen
        inferencias distintas."""
        report = T.cohort_report(_cohort_records([("AAA", 1_000_000, 1.0),
                                                  ("BBB", 1_000_000, 2.0),
                                                  ("CCC", 1_000_000, 3.0)]))
        assert report["companies"] == 3
        assert report["pooled_observations"] == 3
        assert report["assets"]["BTC"]["n_companies"] == 3

    def test_every_rejection_carries_its_reason(self):
        """Una cobertura del 60% sin decir que paso con el otro 40% es indistinguible de un
        filtro mal escrito."""
        records = _cohort_records([("AAA", 1_000_000, 1.0), ("BBB", 1_000_000, 2.0),
                                   ("CCC", 1_000_000, 3.0)])
        records.append(_facts(999, "CLSK", units=1_719, fair_value=100_607_000,
                              assets=2_702_000_000, shares=256_817_073))
        report = T.cohort_report(records)
        assert report["companies_examined"] == 4 and report["companies"] == 3
        assert sum(report["rejections"].values()) == 1
        assert report["rejected"][0]["ticker"] == "CLSK"

    def test_the_thresholds_travel_with_the_data(self):
        policy = T.cohort_report([])["policy"]
        assert policy["treasury_min_asset_share"] == T.TREASURY_MIN_ASSET_SHARE
        assert policy["unit_price_tolerance"] == T.UNIT_PRICE_TOLERANCE
        assert sorted(policy["excluded_sic"]) == sorted(T.EXCLUDED_SIC)
        assert policy["frontier"] == 1.0

    def test_an_empty_archive_reports_zero_instead_of_breaking(self):
        report = T.cohort_report([])
        assert report["companies"] == 0 and report["pooled_observations"] == 0

    def test_the_report_roundtrips(self, tmp_path):
        path = tmp_path / "dat_cohort.json"
        T.write_cohort_report(T.cohort_report([]), path)
        assert T.load_cohort_report(path)["companies"] == 0
        assert T.load_cohort_report(tmp_path / "no-existe.json") is None


# =====================================================================================
# La capa 1: lo poco que se puede ejercitar sin red, ejercitado
# =====================================================================================


class _StubClient:
    """Un cliente que devuelve payloads escritos. Solo para las decisiones de capa 1."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.asked: list[str] = []

    def get_json(self, path, *, params=None, headers=None):
        self.asked.append(path)
        return self.answers.get(path)


class TestNetworkLayerDecisions:
    def _adapter(self, answers):
        return T.DigitalAssetTreasuries(get_source(KEY), client=_StubClient(answers))

    def test_a_spot_trust_is_dropped_before_it_costs_four_requests(self):
        """MEDIDO: 25 de los 138 declarantes son SIC 6221 (iShares Bitcoin Trust, ARK
        21Shares). Un trust crea y redime AL NAV, asi que su mNAV esta clavado en 1 por
        arbitraje y no contiene ninguna informacion sobre venta forzada."""
        path = T.SUBMISSIONS_PATH.format(cik=1980994)
        adapter = self._adapter({path: {"name": "iShares Bitcoin Trust ETF", "sic": "6221",
                                        "tickers": ["IBIT"]}})
        assert adapter._company(1980994) is None

    def test_a_broker_that_custodies_client_crypto_is_dropped_too(self):
        path = T.SUBMISSIONS_PATH.format(cik=1783879)
        adapter = self._adapter({path: {"name": "Robinhood Markets, Inc.", "sic": "6211",
                                        "tickers": ["HOOD"]}})
        assert adapter._company(1783879) is None

    def test_the_common_class_is_the_first_ticker(self):
        """La SEC publica los tickers en el orden del registro y las preferentes van detras.
        MEDIDO: Strategy declara MSTR, STRC, STRD, STRF y STRK."""
        path = T.SUBMISSIONS_PATH.format(cik=1050446)
        adapter = self._adapter({path: {"name": "Strategy Inc", "sic": "6199",
                                        "tickers": ["MSTR", "STRC", "STRD", "STRF", "STRK"]}})
        assert adapter._company(1050446)["ticker"] == "MSTR"

    def test_the_quarters_asked_walk_backwards_across_the_year_boundary(self):
        """Se piden cuatro y no uno porque un trimestre recien cerrado esta a medio llenar:
        MEDIDO 2026-08-13, CY2026Q2I tenia 65 declarantes y CY2026Q1I, 118."""
        from datetime import date

        assert T._quarters_before(date(2026, 2, 15), 3) == ["CY2025Q4I", "CY2025Q3I", "CY2025Q2I"]

    def test_the_quarter_in_progress_is_not_asked_for(self):
        """MEDIDO: el 2026-08-13 estamos en Q3 y `CY2026Q3I` devuelve 200 con cero
        declarantes, porque un frame instantaneo se refiere al ULTIMO dia del trimestre.
        Pedirlo gasta una de las cuatro peticiones en un hueco garantizado."""
        from datetime import date

        assert T._quarters_before(date(2026, 8, 13), 1) == ["CY2026Q2I"]

    def test_a_session_without_close_is_not_a_zero(self):
        """El proveedor manda `null` en los cierres que no existen. Convertirlos en cero
        pondria una capitalizacion de cero y un mNAV de cero, que es la cola inferior."""
        payload = {"chart": {"result": [{
            "timestamp": [1786608000, 1786694400],
            "indicators": {"quote": [{"close": [96.09, None]}]},
        }]}}
        assert T.closes_from_chart(payload) == [("2026-08-13", 96.09)]

    def test_an_empty_chart_is_no_series_and_not_an_exception(self):
        assert T.closes_from_chart({}) == []
        assert T.closes_from_chart({"chart": {"result": []}}) == []


# =====================================================================================
# Como llega a una decision
# =====================================================================================


class TestEncodingAndRadar:
    def test_the_cadence_routes_it_to_the_event_encoding(self):
        source = get_source(KEY)
        assert source.cadence == "event"
        assert source.encoding_kind == ENCODING_EVENT
        assert source.encoding is None  # derivada, no una excepcion declarada

    def test_the_magnitude_is_the_fraction_below_one(self):
        spec = EVENT_SPECS[KEY]
        assert spec.magnitude == "dat_below_nav_share"
        assert spec.magnitude in get_source(KEY).feature_names
        # No se anticipa: lo que tiene fecha conocida es el PLAZO del 10-Q, no su contenido.
        assert spec.announced is False

    def test_a_fattening_lower_tail_reads_as_a_negative_tone(self):
        """Es el unico signo del radar que no necesita una hipotesis sobre el mundo: emitir
        por debajo de 1 diluye, asi que la via barata para levantar caja pasa a ser vender el
        tesoro. Misma direccion, y por el mismo motivo, que `token_unlocks`."""
        assert POLARITY[f"{KEY}{SUFFIX_MAG}"] == -1.0

        frame = pd.DataFrame([
            {ENTITY: "BTC", DAY: pd.Timestamp("2026-08-03", tz="UTC"),
             "dat_below_nav_share": 0.5, "dat_mnav_gap": -0.1, "dat_mnav_p25": 0.6,
             "dat_companies": 12.0, "dat_disclosure_lag_days": 34.0},
        ]).set_index([ENTITY, DAY])
        radar = SignalRadarProvider(
            {KEY: frame},
            HistoricalClock(datetime(2026, 8, 5, tzinfo=UTC)),
            sources=[get_source(KEY)],
        )
        features = radar.features("BTC/USDT")
        assert features["signal_coverage"] > 0
        assert features["signal_tone"] < 0
        assert features["signal_intensity"] > 0

    def test_an_asset_without_a_cohort_reads_as_no_data_and_not_as_calm(self):
        """La trampa de los tres estados: `dat_mnav_mag = 0` significa "ninguna tesoreria
        bajo 1" y tambien "de este activo no se nada". La cobertura es lo que las separa, y
        sin ella una puerta bloquearia por falta de datos."""
        frame = pd.DataFrame([
            {ENTITY: "BTC", DAY: pd.Timestamp("2026-08-03", tz="UTC"),
             "dat_below_nav_share": 0.5, "dat_mnav_gap": -0.1, "dat_mnav_p25": 0.6,
             "dat_companies": 12.0, "dat_disclosure_lag_days": 34.0},
        ]).set_index([ENTITY, DAY])
        radar = SignalRadarProvider(
            {KEY: frame},
            HistoricalClock(datetime(2026, 8, 5, tzinfo=UTC)),
            sources=[get_source(KEY)],
        )
        assert radar.features("DOGE/USDT")["signal_coverage"] == 0.0

    def test_the_four_encoded_columns_are_the_usual_ones(self):
        assert encoded_names(KEY) == (
            f"{KEY}_ahead", f"{KEY}_active", f"{KEY}{SUFFIX_MAG}", f"{KEY}_seen"
        )

    def test_a_stale_cohort_stops_weighing(self):
        """La ventana posterior de `events.py` son diez dias: una foto de la cohorte de hace
        tres meses no describe ninguna distribucion, y el peso cae a cero solo."""
        frame = pd.DataFrame([
            {ENTITY: "BTC", DAY: pd.Timestamp("2026-05-03", tz="UTC"),
             "dat_below_nav_share": 0.5, "dat_mnav_gap": -0.1, "dat_mnav_p25": 0.6,
             "dat_companies": 12.0, "dat_disclosure_lag_days": 34.0},
        ]).set_index([ENTITY, DAY])
        radar = SignalRadarProvider(
            {KEY: frame},
            HistoricalClock(datetime(2026, 8, 5, tzinfo=UTC)),
            sources=[get_source(KEY)],
        )
        assert radar.features("BTC/USDT")["signal_tone"] == 0.0


# =====================================================================================
# El catalogo, visto desde la fuente nueva
# =====================================================================================


class TestCatalogOfTheSource:
    def test_it_is_declared_and_connected(self):
        from ai_trader.signals.adapters import register_all
        from ai_trader.signals.source import REGISTRY

        register_all()
        assert KEY in REGISTRY

    def test_nothing_of_this_source_is_sortable(self):
        """LA VIA UNICA: entra como feature y nada al espacio de busqueda. No es un veto."""
        import json

        from ai_trader.scoring import search_space

        text = json.dumps(getattr(search_space, "SEARCH_SPACE", {}), default=str)
        assert KEY not in text
        for feature in get_source(KEY).feature_names:
            assert feature not in text

    def test_the_declared_lag_is_backed_by_the_measurement(self):
        """La misma disciplina que `history_from` y `typical_adv_usd`: el catalogo copia del
        fichero medido, y si alguien escribe una cifra de memoria, esto falla."""
        row = T.declared_vs_measured_lag()
        if row["declared_days"] is None:
            return  # sin declarar todavia: es un estado honesto, no un fallo
        assert row["measured_days"], "declara retraso y el informe no lo respalda"
        assert row["matches"], (
            f"declara {row['declared_days']} dias y se midieron {row['measured_days']} "
            f"(tolerancia +-{T.LAG_TOLERANCE_DAYS:.0f})"
        )

    def test_a_source_that_declares_a_lag_says_where_it_comes_from(self):
        """None no puede significar dos cosas: "publica en el dia" y "no se ha medido" son
        estados distintos, y el que declara cifra tiene que decir de donde sale."""
        for source in CATALOG:
            if source.disclosure_lag_days is not None:
                assert source.lag_note, f"'{source.key}' declara retraso y no dice de donde"

    def test_the_only_source_with_a_publication_lag_is_this_one(self):
        """Hoy es la unica del catalogo cuyo dato se publica semanas despues de la fecha a
        la que se refiere. El COT tambien tiene desfase y NO declara este campo porque ya lo
        resuelve fechando en el dia de publicacion: el campo existe para el caso en que el
        desfase sobrevive a la derivacion, no para describir todos los desfases."""
        declared = {s.key for s in CATALOG if s.lag_note}
        assert declared <= {KEY}

    def test_the_frame_columns_do_not_collide_with_any_other_source(self):
        names = [f.name for s in CATALOG for f in s.features]
        assert len(names) == len(set(names))

    def test_it_declares_why_it_has_no_adv(self):
        source = get_source(KEY)
        assert source.typical_adv_usd is None
        assert source.adv_note
