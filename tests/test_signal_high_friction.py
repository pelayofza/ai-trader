"""
Tests del LOTE DE ALTA FRICCION: doce fuentes, cero llamadas de red.

Se puede testear sin proveedor por lo mismo que el lote anterior —el puerto de dos capas—
y aqui esa separacion vale mas que nunca: la mitad del trabajo de estas fuentes NO es
descargar, es reconstruir. Encontrar el strike de 25 delta cuando el libro no publica
griegas, agrupar precios de liquidacion en clusters, decidir si un titulo en coreano dice
'alta' o 'baja'. Todo eso vive en la capa pura, y por eso se ejercita aqui con payloads con
la forma EXACTA que devuelven los proveedores, copiada de respuestas medidas el 2026-08-13.

Los tests que valen algo son los que fallarian si alguien "simplificara" una decision
tomada a proposito. Cinco de ellos, que son los cinco sitios donde este lote se puede
romper sin que nada de error:

  - la mediana de apalancamiento se pondera por NOTIONAL y no por numero de posiciones;
  - el IV de 25 delta se INTERPOLA y no se coge del strike mas parecido;
  - un anuncio de 'fin de soporte' contiene la palabra 'soporte' y hay que clasificarlo
    ANTES que un alta;
  - un evento archivado treinta veces (la ventana de captura) cuenta UNA;
  - un mapa de liquidacion caducado NO cubre, aunque tenga la misma pinta que uno fresco.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.observation.signal_radar import (
    POLARITY,
    SignalRadarProvider,
    _price_map_reading,
    _PriceMap,
)
from ai_trader.shared.clock import HistoricalClock
from ai_trader.shared.entities import MARKET_ENTITY
from ai_trader.shared.signals import DAY, ENTITY, OBSERVED
from ai_trader.signals.adapters import attention, deribit, hyperliquid, legal, lending, listings
from ai_trader.signals.adapters.common import unique_records
from ai_trader.signals.catalog import (
    ENCODING_CONTINUOUS,
    ENCODING_EVENT,
    ENCODING_PRICE_MAP,
    CATALOG,
    get_source,
    sources_by_encoding,
)
from ai_trader.signals.events import (
    EVENT_SPECS,
    PRICE_DISTANCE_CAP_PCT,
    PRICE_MAP_STALE_DAYS,
    PRICE_SPECS,
    SUFFIX_MAG,
    SUFFIX_NEAR,
    SUFFIX_SEEN,
    encode_price_arrays,
    encode_price_at,
    is_price_map_source,
    price_encoded_names,
    price_map_encoding_spec,
    price_map_sources,
)
from ai_trader.signals.liquidity import (
    ADV_TOLERANCE_FACTOR,
    SOURCE_VENUE,
    VenueLiquidity,
    declared_vs_measured_adv,
    load_adv_ledger,
    measured_adv,
    write_adv_ledger,
)
from ai_trader.signals.source import raw_record

UTC = timezone.utc


def _record(source_key: str, entity: str, payload, **extra):
    return raw_record(source=source_key, entity=entity, payload=payload, **extra)


# =====================================================================================
# Hyperliquid
# =====================================================================================


class TestHyperliquidSnapshot:
    """La forma de `metaAndAssetCtxs` y del leaderboard, tal cual llegan."""

    META_AND_CTXS = [
        {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
                      {"name": "HYPE", "szDecimals": 2, "maxLeverage": 10}]},
        [
            {"funding": "0.00001131", "openInterest": "40708.11", "dayNtlVlm": "1671186150.16",
             "markPx": "63593.0", "oraclePx": "63621.0", "midPx": "63592.5"},
            {"funding": "-0.00002", "openInterest": "31000.0", "dayNtlVlm": "212143834.0",
             "markPx": "40.5", "oraclePx": "40.4", "midPx": "40.45"},
        ],
    ]

    def test_contexts_pair_universe_and_ctxs_by_position(self):
        out = hyperliquid.asset_contexts(self.META_AND_CTXS)
        assert set(out) == {"BTC", "HYPE"}
        assert out["BTC"]["mark_px"] == 63593.0
        assert out["BTC"]["open_interest_usd"] == pytest.approx(40708.11 * 63593.0)
        assert out["HYPE"]["day_ntl_vlm"] == 212143834.0

    def test_a_mismatched_pair_does_not_shift_every_asset(self):
        """La alineacion es POSICIONAL. Si llegan longitudes distintas, emparejar a ciegas
        pondria el contexto de BTC en HYPE y nadie lo notaria."""
        truncated = [self.META_AND_CTXS[0], self.META_AND_CTXS[1][:1]]
        out = hyperliquid.asset_contexts(truncated)
        assert set(out) == {"BTC"}

    def test_the_leaderboard_comes_unsorted_and_the_sample_is_the_largest(self):
        rows = {"leaderboardRows": [
            {"ethAddress": "0xsmall", "accountValue": "763.0"},
            {"ethAddress": "0xbig", "accountValue": "61956944.55"},
            {"ethAddress": "0xmid", "accountValue": "12628753.0"},
        ]}
        assert hyperliquid.top_accounts(rows, 2) == ["0xbig", "0xmid"]

    def test_a_null_liquidation_price_survives_as_none(self):
        """MEDIDO: el 24% de las posiciones no trae precio de liquidacion. Convertirlo en
        cero pondria un cluster en el precio cero de cada activo."""
        state = {"assetPositions": [
            {"position": {"coin": "BTC", "szi": "4.9", "positionValue": "314620.14",
                          "leverage": {"type": "cross", "value": 3}, "liquidationPx": None}},
            {"position": {"coin": "ETH", "szi": "-100", "positionValue": "180000.0",
                          "leverage": {"type": "isolated", "value": 20},
                          "liquidationPx": "2100.5"}},
        ]}
        rows = hyperliquid.account_positions(state, "0xabc")
        assert [r["coin"] for r in rows] == ["BTC", "ETH"]
        assert rows[0]["liquidation_px"] is None
        assert rows[1]["liquidation_px"] == 2100.5
        assert rows[0]["notional_usd"] == 314620.14


class TestHyperliquidLeverage:
    def _record_for(self, positions):
        return _record(
            "hyperliquid_leverage",
            "BTC",
            {"coin": "BTC", "open_interest_usd": 2.5e9, "sampled_accounts": 200,
             "positions": positions},
            day="2026-08-13",
            request={"series": "leverage"},
        )

    def test_the_median_is_weighted_by_notional_and_not_by_position_count(self):
        """EL TEST QUE PROTEGE LA DECISION. Cien posiciones de mil dolares a 3x y una de
        cincuenta millones a 25x: la mediana simple diria 3 y no describiria nada."""
        positions = [
            {"account": f"0x{i}", "notional_usd": 1_000.0, "leverage": 3.0} for i in range(100)
        ] + [{"account": "0xwhale", "notional_usd": 50_000_000.0, "leverage": 25.0}]
        frame = hyperliquid.HyperliquidLeverage(
            get_source("hyperliquid_leverage")
        ).daily_from_raw([self._record_for(positions)])
        assert frame["hl_leverage_median"].iloc[0] == 25.0

    def test_concentration_groups_by_account_before_ranking(self):
        """Una cuenta con tres posiciones en el mismo activo es UNA mano.

        Sin agrupar, las tres posiciones de `0xa` ocuparian tres de los cinco puestos y la
        concentracion saldria MENOR de lo que es: 100 de 120 en vez de 110 de 120."""
        positions = [
            {"account": "0xa", "notional_usd": 30.0, "leverage": 5.0} for _ in range(3)
        ] + [
            {"account": f"0x{i}", "notional_usd": 5.0, "leverage": 5.0} for i in range(6)
        ]
        frame = hyperliquid.HyperliquidLeverage(
            get_source("hyperliquid_leverage")
        ).daily_from_raw([self._record_for(positions)])
        assert frame["hl_oi_top5_share"].iloc[0] == pytest.approx(110.0 / 120.0)

    def test_high_leverage_share_is_notional_and_uses_the_declared_threshold(self):
        positions = [
            {"account": "0xa", "notional_usd": 75.0, "leverage": hyperliquid.HIGH_LEVERAGE},
            {"account": "0xb", "notional_usd": 25.0, "leverage": 3.0},
        ]
        frame = hyperliquid.HyperliquidLeverage(
            get_source("hyperliquid_leverage")
        ).daily_from_raw([self._record_for(positions)])
        assert frame["hl_high_leverage_share"].iloc[0] == pytest.approx(0.75)


class TestLiquidationClusters:
    """`nearest_cluster` es el nucleo del mapa. Cinco propiedades y todas son decisiones."""

    def test_the_nearest_cluster_wins_even_if_a_bigger_one_is_further(self):
        levels = [(95.0, 100.0), (80.0, 900.0)]  # -5% con 100; -20% con 900
        distance, notional = hyperliquid.nearest_cluster(levels, 100.0, min_share=0.05)
        assert distance == pytest.approx(-4.5)  # centro del cubo [-5, -4)
        assert notional == 100.0

    def test_the_notional_is_cumulative_up_to_the_cluster_on_its_side(self):
        levels = [(99.5, 10.0), (98.5, 10.0), (97.5, 80.0), (110.0, 20.0)]
        distance, notional = hyperliquid.nearest_cluster(levels, 100.0, min_share=0.5)
        # El cluster es el cubo de -3%; el acumulado son los tres de abajo, no el de arriba.
        assert distance == pytest.approx(-2.5)
        assert notional == pytest.approx(100.0)

    def test_a_smooth_map_has_no_cluster_and_that_is_not_a_hole(self):
        """Ningun cubo concentra lo suficiente -> (None, None). Aguas arriba se lee como
        'hay mapa y no hay nada cerca', que es distinto de 'no se nada de este activo'."""
        levels = [(100.0 - i * 0.9, 10.0) for i in range(1, 20)]
        assert hyperliquid.nearest_cluster(levels, 100.0, min_share=0.5) == (None, None)

    def test_levels_beyond_the_search_horizon_do_not_exist(self):
        levels = [(50.0, 1_000_000.0)]  # -50%: MEDIDO, los hay, y no son un cluster cercano
        assert hyperliquid.nearest_cluster(levels, 100.0) == (None, None)

    def test_the_sign_says_which_side_and_the_upside_is_positive(self):
        distance, _ = hyperliquid.nearest_cluster([(104.0, 100.0)], 100.0, min_share=0.5)
        assert distance > 0

    def test_a_level_at_zero_or_negative_price_is_ignored(self):
        assert hyperliquid.nearest_cluster([(0.0, 100.0), (None, 50.0)], 100.0) == (None, None)

    def test_the_frame_publishes_nan_when_there_is_no_cluster(self):
        adapter = hyperliquid.HyperliquidLiquidationMap(get_source("hyperliquid_liqmap"))
        frame = adapter.daily_from_raw([
            _record("hyperliquid_liqmap", "SOL",
                    {"coin": "SOL", "mark_px": 100.0, "positions": 3, "with_liquidation_px": 1,
                     "levels": [[40.0, 1e6]]},
                    day="2026-08-13")
        ])
        assert pd.isna(frame["liq_cluster_distance_pct"].iloc[0])
        assert frame[OBSERVED].iloc[0] == 1  # la observacion existe; el cluster no


# =====================================================================================
# Deribit
# =====================================================================================


class TestDeribitInstruments:
    def test_the_expiry_is_at_eight_utc_and_not_at_midnight(self):
        """Ocho horas sobre un semanal son un 5% del plazo, y el delta lo nota."""
        currency, expiry, strike, kind = deribit.parse_instrument("BTC-25SEP26-84000-P")
        assert (currency, strike, kind) == ("BTC", 84000.0, "P")
        assert expiry == datetime(2026, 9, 25, 8, tzinfo=UTC)

    @pytest.mark.parametrize("name", ["BTC-PERPETUAL", "BTC-25SEP26", "", "ETH-1JAN-100-C"])
    def test_what_is_not_an_option_is_rejected(self, name):
        assert deribit.parse_instrument(name) is None


class TestDeribitDelta:
    def test_at_the_money_is_about_half(self):
        delta = deribit.bs_delta(100.0, 100.0, 0.6, 30 / 365.25, "C")
        assert 0.5 < delta < 0.58  # la deriva de vol lo sube un poco por encima de 0,5

    def test_a_put_is_negative_and_the_pair_differs_by_one(self):
        call = deribit.bs_delta(100.0, 110.0, 0.6, 0.25, "C")
        put = deribit.bs_delta(100.0, 110.0, 0.6, 0.25, "P")
        assert put < 0 < call
        assert call - put == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "args", [(0.0, 100.0, 0.6, 0.25), (100.0, 0.0, 0.6, 0.25),
                 (100.0, 100.0, 0.0, 0.25), (100.0, 100.0, 0.6, 0.0)]
    )
    def test_a_degenerate_input_gives_none_and_not_a_nan(self, args):
        assert deribit.bs_delta(*args, "C") is None

    def test_the_iv_is_interpolated_and_not_snapped_to_the_nearest_strike(self):
        """MEDIDO: en los semanales el candidato mas cercano a 0,25 estaba en 0,187.
        Quedarse con el llamaria 'skew de 25 delta' a un skew de 19 delta."""
        rows = [
            {"delta": 0.187, "iv": 26.89, "kind": "C", "days": 7.0},
            {"delta": 0.289, "iv": 25.88, "kind": "C", "days": 7.0},
        ]
        value = deribit._iv_at_delta(rows, 0.25, "C")
        assert 25.88 < value < 26.89
        assert value != 26.89 and value != 25.88

    def test_outside_the_quoted_range_it_takes_the_edge_instead_of_extrapolating(self):
        rows = [{"delta": 0.40, "iv": 30.0, "kind": "C", "days": 7.0}]
        assert deribit._iv_at_delta(rows, 0.25, "C") == 30.0


class TestVolatilitySurface:
    def _book(self):
        """Un libro con la forma exacta de `get_book_summary_by_currency`, dos vencimientos."""
        rows = []
        for expiry, strikes in (("21AUG26", (60000, 63000, 66000)),
                                ("25SEP26", (55000, 63000, 72000))):
            for strike in strikes:
                for kind, iv in (("C", 30.0 + (strike - 63000) / 1000.0),
                                 ("P", 34.0 - (strike - 63000) / 1000.0)):
                    rows.append({
                        "instrument_name": f"BTC-{expiry}-{strike}-{kind}",
                        "mark_iv": iv,
                        "underlying_price": 63000.0,
                        "open_interest": 100.0,
                    })
        return rows

    def test_the_skew_is_put_minus_call(self):
        out = deribit.volatility_surface(self._book(), as_of="2026-08-13")
        assert out["skew_25d"] > 0  # el libro de prueba pone los puts mas caros
        assert "atm_iv_30d" in out

    def test_an_empty_book_produces_no_columns_instead_of_zeros(self):
        assert deribit.volatility_surface([], as_of="2026-08-13") == {}

    def test_an_expired_book_is_ignored(self):
        """Un vencimiento ya pasado tiene plazo negativo: no produce delta ni columna."""
        assert deribit.volatility_surface(self._book(), as_of="2027-01-01") == {}

    def test_the_dvol_row_takes_the_close_of_the_candle(self):
        adapter = deribit.DeribitVolatility(get_source("deribit_volatility"))
        stamp = int(pd.Timestamp("2026-08-11", tz="UTC").timestamp() * 1000)
        frame = adapter.daily_from_raw([
            _record("deribit_volatility", "BTC", [stamp, 51.88, 52.83, 51.20, 52.69],
                    day="2026-08-11", request={"series": "dvol"})
        ])
        assert frame["dvol_index"].iloc[0] == 52.69


class TestDeribitExpiries:
    def test_open_interest_is_converted_to_dollars_before_being_shared_out(self):
        """El OI llega en CONTRATOS. Un contrato de BTC y uno de ETH no son el mismo dinero,
        asi que sin la conversion la fraccion no seria comparable entre monedas."""
        book = [
            {"instrument_name": "BTC-28AUG26-60000-C", "open_interest": 30.0,
             "underlying_price": 63000.0},
            {"instrument_name": "BTC-25SEP26-70000-C", "open_interest": 70.0,
             "underlying_price": 63000.0},
        ]
        payloads = dict(deribit.expiry_payloads(book, "BTC"))
        assert payloads["2026-08-28"]["oi_usd"] == pytest.approx(30.0 * 63000.0)
        assert payloads["2026-09-25"]["oi_share"] == pytest.approx(0.7)

    def test_the_row_is_dated_at_the_expiry_and_not_at_the_capture(self):
        adapter = deribit.DeribitExpiries(get_source("deribit_expiries"))
        frame = adapter.daily_from_raw([
            _record("deribit_expiries", "BTC",
                    {"currency": "BTC", "expiry": "2026-12-25", "oi_usd": 1e9,
                     "oi_share": 0.3, "strikes": {}},
                    day="2026-12-25", request={"series": "expiry"})
        ])
        assert frame.index.get_level_values(DAY)[0] == pd.Timestamp("2026-12-25", tz="UTC")

    def test_the_same_expiry_captured_thirty_times_is_not_thirty_expiries(self):
        """La captura re-archiva el mismo vencimiento cada dia. Con `sum` el OI se
        multiplicaria por treinta; la politica del catalogo es `last` y por eso gana la
        ultima foto."""
        adapter = deribit.DeribitExpiries(get_source("deribit_expiries"))
        records = [
            _record("deribit_expiries", "BTC",
                    {"currency": "BTC", "expiry": "2026-12-25", "oi_usd": 1e9 + i,
                     "oi_share": 0.3, "strikes": {}},
                    day="2026-12-25", request={"series": "expiry"},
                    fetched_at=datetime(2026, 8, 1 + i, tzinfo=UTC))
            for i in range(3)
        ]
        frame = adapter.daily_from_raw(records)
        assert len(frame) == 1
        assert frame["expiry_oi_usd"].iloc[0] == 1e9 + 2


# =====================================================================================
# El mapa de precios: la tercera codificacion
# =====================================================================================


class TestPriceMapEncoding:
    KEY = "hyperliquid_liqmap"
    SPEC = PRICE_SPECS["hyperliquid_liqmap"]

    def _arrays(self, days, distances, notionals):
        return (
            np.array([np.datetime64(d) for d in days]),
            np.array(distances, dtype=float),
            np.array(notionals, dtype=float),
        )

    def _encode(self, days, distances, notionals, cutoff="2026-08-13"):
        return encode_price_arrays(
            *self._arrays(days, distances, notionals),
            pd.Timestamp(cutoff, tz="UTC"),
            self.KEY,
            self.SPEC,
        )

    def test_proximity_is_measured_in_price_and_is_one_at_the_price(self):
        out = self._encode(["2026-08-12"], [0.0], [1e8])
        assert out[f"{self.KEY}{SUFFIX_NEAR}"] == 1.0

    def test_beyond_the_declared_cap_there_is_nothing_near(self):
        out = self._encode(["2026-08-12"], [PRICE_DISTANCE_CAP_PCT + 1.0], [1e8])
        assert out[f"{self.KEY}{SUFFIX_NEAR}"] == 0.0

    def test_the_magnitude_takes_the_sign_of_the_distance_and_not_of_the_notional(self):
        """Un cluster POR DEBAJO son largos que revientan vendiendo. El notional es una
        cantidad; lo que tiene direccion es donde esta el cluster."""
        below = self._encode(["2026-08-12"], [-5.0], [2e8])
        above = self._encode(["2026-08-12"], [+5.0], [2e8])
        assert below[f"{self.KEY}{SUFFIX_MAG}"] < 0 < above[f"{self.KEY}{SUFFIX_MAG}"]
        assert below[f"{self.KEY}{SUFFIX_MAG}"] == -above[f"{self.KEY}{SUFFIX_MAG}"]

    def test_a_stale_snapshot_encodes_to_nothing(self):
        """La foto de hace dos semanas no describe ningun libro. Ver PRICE_MAP_STALE_DAYS."""
        old = (pd.Timestamp("2026-08-13") - timedelta(days=PRICE_MAP_STALE_DAYS + 5)).date()
        out = self._encode([old.isoformat()], [-2.0], [3e8])
        assert out[f"{self.KEY}{SUFFIX_NEAR}"] == 0.0
        assert out[f"{self.KEY}{SUFFIX_SEEN}"] == 1.0  # hay mapa; lo que no hay es lectura

    def test_a_map_without_cluster_is_seen_but_empty(self):
        out = self._encode(["2026-08-12"], [np.nan], [np.nan])
        assert out[f"{self.KEY}{SUFFIX_SEEN}"] == 1.0
        assert out[f"{self.KEY}{SUFFIX_NEAR}"] == 0.0
        assert out[f"{self.KEY}{SUFFIX_MAG}"] == 0.0

    def test_the_future_is_not_read(self):
        """No hay `_ahead` aqui: un mapa es una foto, y la de manana no existe hoy."""
        out = self._encode(["2026-08-20"], [-1.0], [4e8], cutoff="2026-08-13")
        assert out[f"{self.KEY}{SUFFIX_NEAR}"] == 0.0

    def test_the_cutoff_is_the_same_as_the_bars(self):
        """`encode_price_at` recorta con `visible_cutoff`: el dia de hoy NO se ve."""
        frame = pd.DataFrame(
            [{ENTITY: "BTC", DAY: pd.Timestamp("2026-08-13", tz="UTC"),
              "liq_cluster_distance_pct": -2.0, "liq_cluster_notional_usd": 1e8}]
        ).set_index([ENTITY, DAY])
        out = encode_price_at(frame, self.KEY, datetime(2026, 8, 13, 12, tzinfo=UTC),
                              entities=["BTC"])
        assert out["BTC"][f"{self.KEY}{SUFFIX_NEAR}"] == 0.0

    def test_the_published_columns_are_three_and_there_is_no_wake(self):
        assert price_encoded_names(self.KEY) == (
            f"{self.KEY}{SUFFIX_NEAR}", f"{self.KEY}{SUFFIX_MAG}", f"{self.KEY}{SUFFIX_SEEN}"
        )


class TestEncodingRouting:
    def test_the_cadence_still_decides_by_default(self):
        assert get_source("defillama_hacks").encoding_kind == ENCODING_EVENT
        assert get_source("funding_dispersion").encoding_kind == ENCODING_CONTINUOUS

    def test_only_the_two_maps_declare_the_exception(self):
        declared = {s.key for s in CATALOG if s.encoding is not None}
        assert declared == {"hyperliquid_liqmap", "lending_health"}
        assert {s.key for s in price_map_sources()} == declared

    def test_a_price_map_is_not_an_event_and_has_a_spec(self):
        for source in price_map_sources():
            assert is_price_map_source(source)
            assert source.key in PRICE_SPECS
            assert source.key not in EVENT_SPECS

    def test_every_source_lands_in_exactly_one_encoding(self):
        counted = sum(len(sources_by_encoding(k)) for k in
                      (ENCODING_CONTINUOUS, ENCODING_EVENT, ENCODING_PRICE_MAP))
        assert counted == len(CATALOG)

    def test_the_policy_is_published_with_the_data(self):
        spec = price_map_encoding_spec()
        assert spec["distance_cap_pct"] == PRICE_DISTANCE_CAP_PCT
        assert set(spec["sources"]) == set(PRICE_SPECS)


class TestPriceMapInTheRadar:
    KEY = "hyperliquid_liqmap"

    def _map(self, day: str, distance: float, notional: float) -> _PriceMap:
        return _PriceMap(
            days=np.array([np.datetime64(day)]),
            distances=np.array([distance], dtype=float),
            notionals=np.array([notional], dtype=float),
        )

    def test_a_cluster_below_reads_as_a_negative_tone(self):
        reading = _price_map_reading(
            self.KEY, self._map("2026-08-12", -3.0, 3e8),
            pd.Timestamp("2026-08-13", tz="UTC"), PRICE_SPECS[self.KEY], POLARITY,
        )
        assert reading.covered and reading.tone < 0 and reading.intensity > 0

    def test_a_stale_map_does_not_count_as_coverage(self):
        """La diferencia con una fuente de evento: un calendario viejo sigue siendo cierto;
        una foto vieja del libro no describe ningun libro."""
        reading = _price_map_reading(
            self.KEY, self._map("2026-07-01", -3.0, 3e8),
            pd.Timestamp("2026-08-13", tz="UTC"), PRICE_SPECS[self.KEY], POLARITY,
        )
        assert not reading.covered

    def test_the_radar_indexes_a_map_source_and_publishes_it(self):
        frame = pd.DataFrame([
            {ENTITY: "BTC", DAY: pd.Timestamp(day, tz="UTC"),
             "liq_cluster_distance_pct": -4.0, "liq_cluster_notional_usd": 2e8, OBSERVED: 1}
            for day in ("2026-08-11", "2026-08-12")
        ]).set_index([ENTITY, DAY])

        radar = SignalRadarProvider(
            {self.KEY: frame},
            HistoricalClock(datetime(2026, 8, 13, tzinfo=UTC)),
            sources=[get_source(self.KEY)],
        )
        assert radar.coverage_report()["price_map_sources"] == [self.KEY]
        features = radar.features("BTC/USDT")
        assert features["signal_coverage"] > 0
        assert features["signal_tone"] < 0


# =====================================================================================
# Atencion geografica
# =====================================================================================


class TestAppStore:
    def _chart(self, names):
        return [{"name": name} for name in names]

    def test_visibility_is_one_at_the_top_and_a_hundredth_at_the_bottom(self):
        top = attention.chart_visibility(self._chart(["Upbit"]), ("upbit",))
        assert top == 1.0
        bottom = attention.chart_visibility(
            self._chart([f"App {i}" for i in range(99)] + ["Upbit"]), ("upbit",)
        )
        assert bottom == pytest.approx(0.01)

    def test_out_of_the_chart_is_zero_and_that_is_an_observation(self):
        """MEDIDO 2026-08-13: ninguna de las cuatro apps estaba en el top 100. Cero
        significa 'no esta entre las cien', que se ha observado; NaN significaria que no se
        miro."""
        assert attention.chart_visibility(self._chart(["Banco", "CGV"]), ("upbit",)) == 0.0

    def test_the_best_app_wins_instead_of_adding_them_up(self):
        chart = self._chart(["Coinbase"] + [f"App {i}" for i in range(78)] + ["Binance"])
        assert attention.chart_visibility(chart, ("coinbase", "binance")) == 1.0

    def _records(self, day, kr_names, us_names):
        return [
            _record("appstore_rank", MARKET_ENTITY,
                    {"country": country, "results": self._chart(names)},
                    day=day, request={"country": country})
            for country, names in (("kr", kr_names), ("us", us_names))
        ]

    def test_a_day_with_no_crypto_app_produces_no_row(self):
        """La razon por la que esta fuente es de EVENTO y no continua: una serie de ceros
        no tiene mediana ni rango intercuartilico, y su z no existiria."""
        adapter = attention.AppStoreRank(get_source("appstore_rank"))
        frame = adapter.daily_from_raw(self._records("2026-08-13", ["Banco"], ["ChatGPT"]))
        assert frame.empty

    def test_the_gap_is_korea_minus_the_united_states(self):
        adapter = attention.AppStoreRank(get_source("appstore_rank"))
        frame = adapter.daily_from_raw(
            self._records("2026-08-13", ["Upbit"], [f"App {i}" for i in range(50)] + ["Coinbase"])
        )
        row = frame.iloc[0]
        assert row["app_visibility_kr"] == 1.0
        # Coinbase entra en el puesto 51 de 100 -> visibilidad 0,50.
        assert row["app_visibility_us"] == pytest.approx(0.50)
        assert row["app_visibility_gap"] == pytest.approx(0.50)

    def test_without_both_charts_there_is_no_differential(self):
        adapter = attention.AppStoreRank(get_source("appstore_rank"))
        only_korea = [
            _record("appstore_rank", MARKET_ENTITY,
                    {"country": "kr", "results": self._chart(["Upbit"])},
                    day="2026-08-13", request={"country": "kr"})
        ]
        frame = adapter.daily_from_raw(only_korea)
        assert frame["app_visibility_kr"].iloc[0] == 1.0
        assert pd.isna(frame["app_visibility_gap"].iloc[0])

    def test_the_same_day_captured_twice_is_one_reading(self):
        adapter = attention.AppStoreRank(get_source("appstore_rank"))
        records = self._records("2026-08-13", ["Upbit"], ["Coinbase"])
        frame = adapter.daily_from_raw(records + records)
        assert len(frame) == 1


# =====================================================================================
# Legal
# =====================================================================================


class TestEdgar:
    RESPONSE = {
        "hits": {"total": {"value": 159, "relation": "eq"}, "hits": [{"_source": {}}]},
        "aggregations": {
            "form_filter": {
                "sum_other_doc_count": 0,
                "buckets": [
                    {"key": "13F-HR", "doc_count": 85},
                    {"key": "SC 13G", "doc_count": 5},
                    {"key": "8-K", "doc_count": 69},
                ],
            }
        },
    }

    def test_the_count_comes_from_the_aggregation_and_not_from_the_hits(self):
        """MEDIDO: la respuesta trae como mucho un centenar de hits. Contarlos daria un
        numero bajo justo los dias de vencimiento de 13F, que son los que interesan."""
        counts = legal.form_counts(self.RESPONSE)
        assert counts["total"] == 159
        assert counts["institutional"] == 90
        assert len(self.RESPONSE["hits"]["hits"]) == 1  # y aun asi el total es 159

    def test_a_response_without_total_is_no_data_and_not_a_zero(self):
        assert legal.form_counts({"hits": {}}) is None
        assert legal.form_counts(None) is None

    def test_the_day_window_keeps_the_most_recent_days(self):
        from datetime import date

        days = legal._day_range(date(2014, 1, 1), date(2026, 8, 13), 5)
        assert days == ["2026-08-09", "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"]

    def test_an_inverted_range_is_empty(self):
        from datetime import date

        assert legal._day_range(date(2026, 8, 13), date(2026, 1, 1), 5) == []

    def test_the_row_publishes_both_counts(self):
        adapter = legal.SecEdgarFullText(get_source("sec_edgar_fts"))
        frame = adapter.daily_from_raw([
            _record("sec_edgar_fts", MARKET_ENTITY,
                    {"date": "2026-08-10", "total": 159, "institutional": 90,
                     "by_form": {}, "other": 0},
                    day="2026-08-10")
        ])
        assert frame["edgar_filings"].iloc[0] == 159.0
        assert frame["edgar_institutional"].iloc[0] == 90.0


class TestFederalRegister:
    def _doc(self, number, day, kind):
        return _record("federal_register", MARKET_ENTITY,
                       {"document_number": number, "publication_date": day, "type": kind},
                       day=day)

    def test_only_rules_count_as_regulatory_activity(self):
        adapter = legal.FederalRegister(get_source("federal_register"))
        frame = adapter.daily_from_raw([
            self._doc("2026-1", "2026-08-12", "Proposed Rule"),
            self._doc("2026-2", "2026-08-12", "Notice"),
        ])
        assert frame["fedreg_documents"].iloc[0] == 2.0
        assert frame["fedreg_rules"].iloc[0] == 1.0

    def test_the_same_document_archived_thirty_times_counts_once(self):
        """EL TEST QUE PROTEGE LA COLUMNA SUMADA. La captura pide treinta dias y re-archiva
        lo mismo cada dia; sin deduplicar, un documento seria un mes muy activo."""
        adapter = legal.FederalRegister(get_source("federal_register"))
        copies = [
            _record("federal_register", MARKET_ENTITY,
                    {"document_number": "2026-16460", "publication_date": "2026-08-12",
                     "type": "Notice"},
                    day="2026-08-12", fetched_at=datetime(2026, 8, 12 + i % 15, tzinfo=UTC))
            for i in range(30)
        ]
        frame = adapter.daily_from_raw(copies)
        assert frame["fedreg_documents"].iloc[0] == 1.0


class TestCourtListener:
    def test_dockets_are_counted_once_per_docket_id(self):
        adapter = legal.CourtListenerDockets(get_source("courtlistener_dockets"))
        records = [
            _record("courtlistener_dockets", MARKET_ENTITY,
                    {"docket_id": 74638444, "dateFiled": "2026-08-12", "court_id": "paeb"},
                    day="2026-08-12"),
            _record("courtlistener_dockets", MARKET_ENTITY,
                    {"docket_id": 74638444, "dateFiled": "2026-08-12", "court_id": "paeb"},
                    day="2026-08-12"),
            _record("courtlistener_dockets", MARKET_ENTITY,
                    {"docket_id": 999, "dateFiled": "2026-08-12", "court_id": "cand"},
                    day="2026-08-12"),
        ]
        assert adapter.daily_from_raw(records)["court_dockets"].iloc[0] == 2.0

    def test_the_cursor_is_read_from_the_next_url(self):
        assert legal._cursor_of("https://x/api?q=a&cursor=cD0yMDI2") == "cD0yMDI2"
        assert legal._cursor_of(None) is None
        assert legal._cursor_of("https://x/api?q=a") is None


# =====================================================================================
# Listados y deslistados
# =====================================================================================


class TestUpbitTitles:
    """Los titulos son los MEDIDOS el 2026-08-13, copiados tal cual."""

    def test_a_new_asset_is_a_listing(self):
        assert listings.classify("프롬(PROM) KRW, USDT 마켓 디지털 자산 추가") == listings.LISTING_ADDED

    def test_the_two_thousand_seventeen_format_still_parses(self):
        assert listings.classify("이더리움클래식(ETC/KRW) 상장") == listings.LISTING_ADDED

    def test_the_end_of_support_is_a_delisting_and_not_a_listing(self):
        """EL ORDEN IMPORTA: '거래지원 종료' contiene '거래지원', que marca un alta. Probando
        el alta primero, TODAS las bajas se clasificarian como altas."""
        assert listings.classify("비트토렌트(BTT) 거래지원 종료 안내") == listings.LISTING_REMOVED

    def test_the_warning_designation_is_its_own_class(self):
        assert listings.classify("레이븐코인(RVN) 거래 유의 종목 지정 안내") == listings.LISTING_WARNED

    def test_an_unrelated_announcement_is_not_forced_into_a_class(self):
        assert listings.classify("서버 점검 안내") is None

    def test_the_ticker_comes_from_the_parentheses(self):
        assert listings.tickers("프롬(PROM) KRW, USDT 마켓 디지털 자산 추가") == ("PROM",)
        assert listings.tickers("이더리움클래식(ETC/KRW) 상장") == ("ETC",)

    def test_a_multi_token_announcement_names_them_all(self):
        title = "BTC, USDT 마켓 신규 거래지원 안내 (CYS, ICNT, XAN, EDEN, AIOZ, ALLO)"
        assert listings.tickers(title) == ("CYS", "ICNT", "XAN", "EDEN", "AIOZ", "ALLO")

    def test_a_market_list_is_not_a_token_list(self):
        """'(KRW, BTC, USDT 마켓)' son los mercados donde se lista, no tres listados."""
        assert listings.tickers("댑오에스(DOS) 신규 거래지원 안내 (KRW, BTC, USDT 마켓)") == ("DOS",)

    def test_the_korean_timestamp_becomes_a_utc_day(self):
        moment = listings.announcement_moment({"listed_at": "2026-08-12T11:11:38+09:00"})
        assert moment == datetime(2026, 8, 12, 2, 11, 38, tzinfo=UTC)

    def test_an_early_morning_announcement_belongs_to_the_previous_utc_day(self):
        moment = listings.announcement_moment({"listed_at": "2026-08-12T08:00:00+09:00"})
        assert moment.date().isoformat() == "2026-08-11"


class TestUpbitFrame:
    def _notice(self, notice_id, title, stamp="2026-08-12T11:11:38+09:00"):
        return _record("cex_listings", "upbit",
                       {"id": notice_id, "title": title, "listed_at": stamp},
                       day="2026-08-12")

    def test_a_listing_is_plus_one_and_a_delisting_is_minus_one(self):
        adapter = listings.UpbitListings(get_source("cex_listings"))
        frame = adapter.daily_from_raw([
            self._notice(1, "프롬(PROM) KRW, USDT 마켓 디지털 자산 추가"),
            self._notice(2, "비트토렌트(BTT) 거래지원 종료 안내"),
        ])
        assert frame.loc[("PROM",), "listing_change"].iloc[0] == 1.0
        assert frame.loc[("BTT",), "listing_change"].iloc[0] == -1.0

    def test_a_warning_is_not_half_a_delisting(self):
        adapter = listings.UpbitListings(get_source("cex_listings"))
        frame = adapter.daily_from_raw([
            self._notice(3, "레이븐코인(RVN) 거래 유의 종목 지정 안내")
        ])
        row = frame.loc[("RVN",)].iloc[0]
        assert row["listing_change"] == 0.0
        assert row["listing_warning"] == 1.0

    def test_the_same_announcement_captured_daily_counts_once(self):
        adapter = listings.UpbitListings(get_source("cex_listings"))
        copies = [
            _record("cex_listings", "upbit",
                    {"id": 6466, "title": "프롬(PROM) KRW, USDT 마켓 디지털 자산 추가",
                     "listed_at": "2026-08-12T11:11:38+09:00"},
                    day="2026-08-12", fetched_at=datetime(2026, 8, 12 + i, tzinfo=UTC))
            for i in range(10)
        ]
        frame = adapter.daily_from_raw(copies)
        assert frame["listing_change"].iloc[0] == 1.0

    def test_one_announcement_with_six_tokens_is_six_events(self):
        adapter = listings.UpbitListings(get_source("cex_listings"))
        frame = adapter.daily_from_raw([
            self._notice(4, "BTC, USDT 마켓 신규 거래지원 안내 (CYS, ICNT, XAN, EDEN, AIOZ, ALLO)")
        ])
        assert len(frame) == 6
        assert set(frame.index.get_level_values(ENTITY)) == {
            "CYS", "ICNT", "XAN", "EDEN", "AIOZ", "ALLO"
        }


# =====================================================================================
# Prestamos on-chain: lo que se midio es que no hay via gratuita
# =====================================================================================


class TestLending:
    def test_a_two_hundred_with_an_error_inside_is_an_error(self):
        """MEDIDO: el gateway responde 200 y mete el fallo en el cuerpo. Sin mirar dentro,
        la ingesta archivaria una respuesta vacia con pinta de exito."""
        payload = {"errors": [{"message": "auth error: missing authorization header"}]}
        assert lending.graphql_error(payload) == "auth error: missing authorization header"
        assert lending.graphql_error({"data": {"accounts": []}}) is None

    def test_without_a_credential_it_raises_instead_of_returning_nothing(self, monkeypatch):
        monkeypatch.delenv("THEGRAPH_API_KEY", raising=False)
        adapter = lending.LendingHealth(get_source("lending_health"))
        with pytest.raises(RuntimeError, match="THEGRAPH_API_KEY"):
            adapter.fetch_raw(["ETH"])

    def test_a_healthy_loan_far_from_liquidation_is_not_in_the_map(self):
        payload = {"data": {"accounts": [
            {"id": "0xa", "positions": [
                {"side": "COLLATERAL", "balance": 1000.0, "asset": {"symbol": "WETH"}},
                {"side": "BORROWER", "balance": 100.0, "asset": {"symbol": "USDC"}},
            ]},
        ]}}
        # health factor 10 -> por encima de MAX_HEALTH_FACTOR: ese prestamo no describe
        # el riesgo de nadie y no aporta nivel.
        assert lending.collateral_blocks(payload, lending.COLLATERAL) == {}

    def test_a_stressed_loan_becomes_a_signed_level(self):
        payload = {"data": {"accounts": [
            {"id": "0xa", "positions": [
                {"side": "COLLATERAL", "balance": 125.0, "asset": {"symbol": "WETH"}},
                {"side": "BORROWER", "balance": 100.0, "asset": {"symbol": "USDC"}},
            ]},
        ]}}
        blocks = lending.collateral_blocks(payload, lending.COLLATERAL)
        drop, notional = blocks["WETH"]["levels"][0]
        assert drop == pytest.approx(-20.0)  # health factor 1,25 aguanta una caida del 20%
        assert notional == 125.0

    def test_the_map_uses_the_same_algorithm_as_the_perpetual(self):
        """No hay dos implementaciones del mapa: `lending` importa la de `hyperliquid`."""
        record = _record("lending_health", "ETH",
                         {"subgraph": "aave-v3-ethereum", "symbol": "WETH",
                          # El cubo del -4% no llega al umbral y el del -4,5% si: el cluster
                          # es el segundo, y el notional que se publica los acumula los dos.
                          "levels": [[-4.0, 1e8], [-4.5, 2.9e9]], "loans": 2},
                         day="2026-08-13")
        from ai_trader.signals.adapters.lending import _lending_row

        row = _lending_row(record)
        assert row["lending_liq_distance_pct"] == pytest.approx(-4.5)
        assert row["lending_liq_notional_usd"] == pytest.approx(3e9)


# =====================================================================================
# Deduplicacion: la pieza comun que protege las columnas sumadas
# =====================================================================================


class TestUniqueRecords:
    def test_the_last_capture_of_a_fact_wins(self):
        records = [
            _record("x", "*", {"id": 1, "v": "viejo"}, fetched_at=datetime(2026, 8, 1, tzinfo=UTC)),
            _record("x", "*", {"id": 1, "v": "nuevo"}, fetched_at=datetime(2026, 8, 5, tzinfo=UTC)),
        ]
        out = unique_records(records, key_of=lambda r: r["payload"]["id"])
        assert len(out) == 1 and out[0]["payload"]["v"] == "nuevo"

    def test_the_order_of_the_archive_does_not_change_the_answer(self):
        records = [
            _record("x", "*", {"id": 1, "v": i}, fetched_at=datetime(2026, 8, 1 + i, tzinfo=UTC))
            for i in range(5)
        ]
        forward = unique_records(records, key_of=lambda r: r["payload"]["id"])
        backward = unique_records(list(reversed(records)), key_of=lambda r: r["payload"]["id"])
        assert forward[0]["payload"]["v"] == backward[0]["payload"]["v"] == 4

    def test_a_record_without_identity_is_dropped_and_not_merged(self):
        records = [_record("x", "*", {"v": 1}), _record("x", "*", {"v": 2})]
        assert unique_records(records, key_of=lambda r: r["payload"].get("id")) == []


# =====================================================================================
# El ADV: la cifra que hay que tener delante antes de escalar
# =====================================================================================


class TestDeclaredLiquidity:
    def test_every_declared_adv_is_backed_by_a_measurement(self):
        """La misma disciplina que `history_from`: el catalogo copia del registro, y si
        alguien escribe una cifra de memoria, esto falla."""
        for key, row in declared_vs_measured_adv().items():
            if row["declared_usd"] is None:
                continue
            assert row["measured_usd"], f"'{key}' declara ADV y el registro no lo respalda"
            assert row["within_tolerance"], (
                f"'{key}' declara {row['declared_usd']:,.0f} y se midio "
                f"{row['measured_usd']:,.0f} (tolerancia x{ADV_TOLERANCE_FACTOR:.0f})"
            )

    def test_a_source_with_a_measurable_venue_declares_it(self):
        for key, venue in SOURCE_VENUE.items():
            source = get_source(key)
            measured = measured_adv().get(key)
            if measured:
                assert source.typical_adv_usd, (
                    f"'{key}' tiene ADV medido en {venue} y el catalogo no lo declara"
                )

    def test_market_scoped_sources_declare_why_they_have_no_adv(self):
        """None no puede significar dos cosas. En las de alcance mercado el eje no es un
        activo, y eso se escribe."""
        for source in CATALOG:
            if source.typical_adv_usd is None and source.key in {
                "appstore_rank", "naver_datalab", "yandex_wordstat",
                "sec_edgar_fts", "federal_register", "courtlistener_dockets",
                "lending_health",
            }:
                assert source.adv_note, f"'{source.key}' no dice por que no tiene ADV"

    def test_the_thinnest_declared_source_is_visible_in_the_summary(self):
        from ai_trader.signals.liquidity import liquidity_summary

        summary = liquidity_summary()
        declared = [s.typical_adv_usd for s in CATALOG if s.typical_adv_usd]
        assert summary["thinnest_adv_usd"] == min(declared)

    def test_the_ledger_roundtrips(self, tmp_path):
        path = tmp_path / "entity_adv.json"
        measured = {
            "venue_x": VenueLiquidity(
                venue="venue_x", measured_at="2026-08-13T00:00:00+00:00",
                entities={"AAA": 1000.0, "BBB": 0.0, "CCC": 3000.0},
            )
        }
        write_adv_ledger(measured, path, merge=False)
        row = load_adv_ledger(path)["venues"]["venue_x"]
        assert row["n_entities"] == 3
        assert row["n_traded"] == 2  # el cero se conserva y no entra en la mediana
        assert row["median_usd"] == 2000.0

    def test_a_partial_measurement_does_not_erase_the_rest(self, tmp_path):
        path = tmp_path / "entity_adv.json"
        write_adv_ledger(
            {"a": VenueLiquidity("a", "2026-08-01T00:00:00+00:00", {"X": 1.0})}, path, merge=False
        )
        write_adv_ledger(
            {"b": VenueLiquidity("b", "2026-08-13T00:00:00+00:00", {"Y": 2.0})}, path
        )
        assert set(load_adv_ledger(path)["venues"]) == {"a", "b"}


# =====================================================================================
# El catalogo, visto desde el lote nuevo
# =====================================================================================


class TestCatalogOfTheBatch:
    NEW = (
        "hyperliquid_leverage", "hyperliquid_liqmap", "deribit_volatility", "deribit_expiries",
        "lending_health", "appstore_rank", "naver_datalab", "yandex_wordstat",
        "sec_edgar_fts", "federal_register", "courtlistener_dockets", "cex_listings",
    )

    def test_the_twelve_are_declared_and_connected(self):
        from ai_trader.signals.adapters import register_all
        from ai_trader.signals.source import REGISTRY

        register_all()
        for key in self.NEW:
            assert get_source(key), key
            assert key in REGISTRY, f"'{key}' esta declarada y no tiene adaptador"

    def test_nothing_of_this_batch_is_sortable(self):
        """LA VIA UNICA: todo entra como feature y nada al espacio de busqueda."""
        from ai_trader.scoring import search_space

        text = json.dumps(getattr(search_space, "SEARCH_SPACE", {}), default=str)
        for key in self.NEW:
            assert key not in text
        for source in CATALOG:
            for feature in source.feature_names:
                assert feature not in text

    def test_a_source_that_could_not_be_measured_keeps_history_from_none(self):
        """Tres piden credencial y una no tiene pasado: ninguna puede entrar en un backtest,
        y eso es la medicion y no un pendiente."""
        for key in ("lending_health", "naver_datalab", "yandex_wordstat",
                    "hyperliquid_leverage", "hyperliquid_liqmap"):
            assert get_source(key).history_from is None
            assert not get_source(key).backtestable

    def test_the_measured_ones_declare_what_the_probe_found(self):
        from ai_trader.signals.depth import declared_vs_measured

        comparison = declared_vs_measured()
        for key in ("deribit_volatility", "sec_edgar_fts", "federal_register", "cex_listings"):
            row = comparison[key]
            assert row["declared"] == row["declarable"], f"'{key}': {row}"

    def test_every_new_feature_name_is_unique_in_the_catalog(self):
        names = [f.name for s in CATALOG for f in s.features]
        assert len(names) == len(set(names))
