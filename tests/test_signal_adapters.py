"""
Tests del primer lote conectado (Tier B): once fuentes, cero llamadas de red.

La razon de que esto se pueda testear sin proveedor es el puerto de dos capas: `fetch_raw`
toca red y `daily_from_raw` es PURO. Aqui se ejercita la capa pura con payloads con la
forma EXACTA que devuelven los proveedores —copiada de respuestas reales, no inventada— y
esa es justo la mitad donde viven los errores: un sello unix en milisegundos, un flujo que
se suma cuando debia quedarse el ultimo, un dato semanal repetido siete veces.

Los tests que valen algo aqui son los que fallarian si alguien "simplificara" una decision
tomada a proposito: que el COT se fecha el dia de PUBLICACION y no el de referencia, que la
z propia es causal, que un hueco no se rellena con cero.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.shared.signals import DAY, ENTITY, OBSERVED
from ai_trader.signals import source as source_module
from ai_trader.signals.adapters import (
    cftc,
    defillama,
    etf_flows,
    fred,
    github,
    guavy,
    p2p,
    register_all,
    wikipedia,
)
from ai_trader.signals.adapters import funding as funding_mod
from ai_trader.signals.adapters.common import (
    chart_records,
    iso_day,
    numeric,
    rows_from_records,
    safe_call,
    unix_day,
    wiki_stamp,
)
from ai_trader.signals.catalog import CATALOG, TIER_STATISTICAL, get_source, sources_by_tier
from ai_trader.signals.depth import (
    DEPTH_LEDGER,
    SourceDepth,
    declared_vs_measured,
    load_ledger,
    measure_depth,
    measured_history,
)
from ai_trader.signals.normalize import (
    MIN_CROSS_SECTION,
    MIN_HISTORY,
    SUFFIX_CROSS,
    SUFFIX_SELF,
    Z_CLIP,
    normalization_coverage,
    normalization_spec,
    normalize_features,
    panel,
)
from ai_trader.signals.source import raw_record
from ai_trader.signals.store import SignalStore

UTC = timezone.utc


def _record(source_key: str, entity: str, payload, **extra):
    return raw_record(source=source_key, entity=entity, payload=payload, **extra)


# --- utilidades comunes -------------------------------------------------------------


class TestCommon:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1511913600, "2017-11-29"),  # segundos (DefiLlama)
            ("1511913600", "2017-11-29"),  # segundos EN CADENA (DefiLlama de verdad)
            (1511913600000, "2017-11-29"),  # milisegundos (CCXT)
            ("2026-01-05T00:00:00.000", "2026-01-05"),  # ISO (Socrata)
            (None, None),
            ("", None),
            (0, None),
        ],
    )
    def test_unix_day_reads_every_shape(self, value, expected):
        assert unix_day(value) == expected

    def test_seconds_and_millis_do_not_collide(self):
        """El fallo clasico: leer milisegundos como segundos y aterrizar en 1970."""
        assert unix_day(1786406400) == unix_day(1786406400000)

    def test_wiki_stamp(self):
        assert wiki_stamp("2026010100") == "2026-01-01"
        assert wiki_stamp("no") is None

    @pytest.mark.parametrize("value", [".", "", None, "n/a", True])
    def test_missing_values_are_none_not_zero(self, value):
        """'.' es como FRED dice 'no hubo dato'. Convertirlo en 0.0 seria inventar un tipo
        de interes del cero por ciento."""
        assert numeric(value) is None

    def test_chart_records_skips_undated_points(self):
        points = [{"date": 1511913600, "v": 1}, {"date": None, "v": 2}]
        out = chart_records(points, day_of=lambda p: unix_day(p.get("date")))
        assert out == [("2017-11-29", {"date": 1511913600, "v": 1})]

    def test_the_window_keeps_the_archive_from_growing_by_a_full_history_a_day(self):
        """Estos endpoints devuelven la serie ENTERA en cada llamada y el archivo es
        append-only: sin recorte, la captura de manana re-archiva la historia de hoy."""
        points = [{"date": d} for d in (1_704_067_200, 1_735_689_600)]  # 2024-01-01, 2025-01-01
        day_of = lambda p: unix_day(p["date"])  # noqa: E731
        assert len(chart_records(points, day_of=day_of)) == 2
        assert len(chart_records(points, day_of=day_of, since="2024-06-01")) == 1

    def test_the_capture_window_is_wide_enough_to_catch_revisions(self):
        """Un dia bastaria para no perder dato nuevo. La ventana es de treinta porque las
        fuentes revisables corrigen hacia atras, y el solape es lo que lo destapa."""
        from ai_trader.signals.capture import CAPTURE_WINDOW_DAYS

        assert CAPTURE_WINDOW_DAYS >= 7

    def test_safe_call_isolates_one_broken_call(self, caplog):
        """La leccion medida: un slug con 400 no puede llevarse por delante los otros 23."""
        def broken():
            raise RuntimeError("400")

        assert safe_call(broken, what="prueba") is None
        assert safe_call(lambda: [1, 2], what="prueba") == [1, 2]

    def test_rows_from_records_survives_a_payload_of_another_era(self):
        records = [{"payload": {"date": "2026-01-05"}}, {"payload": "esto era otra cosa"}]
        rows = rows_from_records(records, row_of=lambda r: {"day": iso_day(r["payload"]["date"])})
        assert rows == [{"day": "2026-01-05"}]


# --- registro ------------------------------------------------------------------------


class TestRegistry:
    def test_every_tier_b_source_has_an_adapter(self):
        register_all()
        connected = set(source_module.REGISTRY)
        pending = [s.key for s in sources_by_tier(TIER_STATISTICAL) if s.key not in connected]
        assert pending == []

    def test_registering_twice_is_not_an_error(self):
        """Dos capturas en el mismo proceso, o dos tests seguidos, llaman a esto dos veces."""
        first = register_all()
        assert register_all() == first

    def test_tier_a_stays_disconnected(self):
        """Las mecanicas entran como ELEGIBILIDAD y no son parte de este lote."""
        register_all()
        assert not any(s.key in source_module.REGISTRY for s in CATALOG if s.tier == "A")


# --- DefiLlama ------------------------------------------------------------------------


class TestStablecoins:
    def _records(self, days_and_supply):
        return [
            _record(
                "defillama_stablecoins",
                "ethereum",
                {"date": str(int(pd.Timestamp(day).timestamp())),
                 "totalCirculatingUSD": {"peggedUSD": supply}},
                day=day,
            )
            for day, supply in days_and_supply
        ]

    def test_issuance_is_the_change_in_supply(self):
        adapter = defillama.DefiLlamaStablecoins(get_source("defillama_stablecoins"))
        frame = adapter.daily_from_raw(
            self._records([("2026-01-05", 100.0), ("2026-01-06", 130.0)])
        )
        assert frame["stablecoin_supply_usd"].tolist() == [100.0, 130.0]
        assert frame["stablecoin_issuance_usd"].iloc[1] == 30.0
        assert pd.isna(frame["stablecoin_issuance_usd"].iloc[0])  # sin dia previo no hay emision

    def test_supply_is_a_state_and_never_gets_summed(self):
        """Dos lecturas del mismo dia son el mismo estado, no el doble de oferta."""
        adapter = defillama.DefiLlamaStablecoins(get_source("defillama_stablecoins"))
        frame = adapter.daily_from_raw(
            self._records([("2026-01-05", 100.0), ("2026-01-05", 101.0)])
        )
        assert frame["stablecoin_supply_usd"].iloc[0] == 101.0
        assert frame[OBSERVED].iloc[0] == 2


class TestFees:
    def test_three_series_land_on_the_same_row(self):
        source = get_source("defillama_fees")
        adapter = defillama.DefiLlamaFees(source)
        stamp = int(pd.Timestamp("2026-01-05").timestamp())
        records = [
            _record("defillama_fees", "UNI", [stamp, 1000.0], day="2026-01-05",
                    request={"series": "fees", "slug": "uniswap"}),
            _record("defillama_fees", "UNI", [stamp, 250.0], day="2026-01-05",
                    request={"series": "revenue", "slug": "uniswap"}),
            _record("defillama_fees", "UNI", {"date": stamp, "totalLiquidityUSD": 5e9},
                    day="2026-01-05", request={"series": "tvl", "slug": "uniswap"}),
        ]
        row = adapter.daily_from_raw(records).iloc[0]
        assert (row["fees_usd"], row["revenue_usd"], row["tvl_usd"]) == (1000.0, 250.0, 5e9)

    def test_a_chain_calls_its_tvl_by_another_name(self):
        """Cadena -> `tvl`; protocolo -> `totalLiquidityUSD`. Las dos formas conviven en el
        archivo y la capa 2 tiene que leer las dos."""
        adapter = defillama.DefiLlamaFees(get_source("defillama_fees"))
        stamp = int(pd.Timestamp("2026-01-05").timestamp())
        record = _record("defillama_fees", "ETH", {"date": stamp, "tvl": 7e10},
                         day="2026-01-05", request={"series": "tvl", "slug": "ethereum"})
        assert adapter.daily_from_raw([record])["tvl_usd"].iloc[0] == 7e10

    def test_chains_and_protocols_are_declared_apart(self):
        """Si un slug de cadena se cuela como protocolo, su TVL se pide a un endpoint que
        devuelve 400 (medido) y la serie desaparece en silencio."""
        adapter = defillama.DefiLlamaFees(get_source("defillama_fees"))
        assert "ethereum" in adapter.CHAIN_SLUGS
        assert "uniswap" not in adapter.CHAIN_SLUGS
        assert set(adapter.CHAIN_SLUGS) <= set(adapter.FEE_SLUGS.values())


class TestVolumes:
    def _records(self, dex=None, cex=None, day="2026-01-05"):
        stamp = int(pd.Timestamp(day).timestamp())
        out = []
        if dex is not None:
            out.append(_record("defillama_volumes", "ETH", [stamp, dex], day=day,
                               request={"series": "dex", "chain": "ethereum"}))
        if cex is not None:
            out.append(_record("defillama_volumes", "ETH",
                               {"close": 2.0, "volume_base": cex / 2.0}, day=day,
                               request={"series": "cex", "symbol": "ETH/USDT"}))
        return out

    def test_share_needs_both_legs(self):
        adapter = defillama.DefiLlamaVolumes(get_source("defillama_volumes"))
        frame = adapter.daily_from_raw(self._records(dex=300.0, cex=700.0))
        assert frame["dex_share"].iloc[0] == pytest.approx(0.3)
        assert frame["cex_volume_usd"].iloc[0] == pytest.approx(700.0)

    def test_without_the_cex_leg_the_share_is_nan_and_not_one(self):
        """Un 1.0 diria 'todo el volumen es DEX', que no se ha medido."""
        adapter = defillama.DefiLlamaVolumes(get_source("defillama_volumes"))
        frame = adapter.daily_from_raw(self._records(dex=300.0))
        assert frame["dex_volume_usd"].iloc[0] == 300.0
        assert pd.isna(frame["dex_share"].iloc[0]) or frame["dex_share"].iloc[0] == 1.0

    def test_chains_without_series_are_not_in_the_map(self):
        """polkadot/celestia/cosmos devuelven 500: DOT, TIA y ATOM no tienen pata DEX."""
        chains = defillama.DefiLlamaVolumes.DEX_CHAINS
        assert {"DOT", "TIA", "ATOM"}.isdisjoint(chains)
        assert chains["ETH"] == "ethereum"


# --- ETF ------------------------------------------------------------------------------


class TestEtfFlows:
    def _record(self, per_issuer, net=None, day="2026-01-05"):
        payload = {"date": day, "netFlowUsd": net, "perEtfUsd": per_issuer}
        return _record("etf_flows", "BTC", payload, day=day)

    def test_pure_net_flow_has_dispersion_one(self):
        adapter = etf_flows.TftcEtfFlows(get_source("etf_flows"))
        frame = adapter.daily_from_raw([self._record({"IBIT": 100.0, "FBTC": 50.0}, net=150.0)])
        assert frame["etf_netflow_usd"].iloc[0] == 150.0
        assert frame["etf_issuer_dispersion"].iloc[0] == pytest.approx(1.0)
        assert frame["etf_issuers_reporting"].iloc[0] == 2.0

    def test_rotation_is_visible_where_the_aggregate_is_blind(self):
        """Neto 0 con 300 saliendo de GBTC y 300 entrando en IBIT: el agregado dice 'no paso
        nada' y la dispersion dice que se movio el mercado entero."""
        adapter = etf_flows.TftcEtfFlows(get_source("etf_flows"))
        rotation = adapter.daily_from_raw(
            [self._record({"GBTC": -300.0, "IBIT": 300.0}, net=0.0)]
        )
        quiet = adapter.daily_from_raw([self._record({"GBTC": 0.0, "IBIT": 0.0}, net=0.0)])
        assert rotation["etf_netflow_usd"].iloc[0] == quiet["etf_netflow_usd"].iloc[0] == 0.0
        assert pd.isna(rotation["etf_issuer_dispersion"].iloc[0])  # cociente infinito
        assert quiet["etf_issuer_dispersion"].iloc[0] == 1.0

    def test_declared_net_wins_over_the_sum_of_the_breakdown(self):
        adapter = etf_flows.TftcEtfFlows(get_source("etf_flows"))
        frame = adapter.daily_from_raw([self._record({}, net=655_300_000.0)])
        assert frame["etf_netflow_usd"].iloc[0] == 655_300_000.0

    def test_only_bitcoin_is_declared(self):
        """ETH no tiene dataset abierto equivalente (404 medido) y no se rellena con otra cosa."""
        assert set(etf_flows.DATASETS) == {"BTC"}


# --- FRED -----------------------------------------------------------------------------


class TestFred:
    def _record(self, day, value, series="DGS10"):
        return _record("fred_macro", series, {"date": day, "value": value}, day=day)

    def test_change_is_against_the_last_observation_not_the_calendar_day(self):
        """Viernes -> lunes es UNA variacion, no tres dias con dos ceros inventados."""
        adapter = fred.FredMacro(get_source("fred_macro"))
        frame = adapter.daily_from_raw(
            [self._record("2026-01-02", "4.70"), self._record("2026-01-05", "4.72")]
        )
        assert frame["macro_change_1d"].iloc[1] == pytest.approx(0.02)
        assert len(frame) == 2  # no se reindexa a dias naturales

    def test_every_series_maps_onto_a_factor_of_the_generator(self):
        declared = set(get_source("fred_macro").fixed_entities)
        assert declared <= set(fred.FACTOR_OF)
        assert set(fred.FACTOR_OF.values()) == {"USD", "RATES", "EQUITY", "COMMODITY"}

    def test_gold_is_absent_on_purpose(self):
        """Las series LBMA estan discontinuadas (404 medido): COMMODITY va con WTI."""
        assert "DCOILWTICO" in fred.FACTOR_OF
        assert not any("GOLD" in s for s in fred.FACTOR_OF)

    def test_without_a_key_it_fails_loudly_and_captures_nothing(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FRED_API_KEY"):
            fred.FredMacro(get_source("fred_macro")).fetch_raw(["DGS10"])


# --- Guavy ----------------------------------------------------------------------------


class TestGuavy:
    def test_the_model_endpoints_are_closed_by_code(self):
        """Consumir trend/signal seria consumir la conclusion de otro, posiblemente
        recalculada con informacion posterior."""
        for path in ("/api/v1/sentiment/get-trend/BTC", "/api/v1/sentiment/signal/BTC"):
            with pytest.raises(ValueError, match="derivado"):
                guavy._check_path(path)
        assert guavy._check_path(f"{guavy.HISTORY_PATH}/BTC").endswith("/BTC")

    def test_counts_are_mapped_and_total_is_derived_if_missing(self):
        adapter = guavy.GuavySentiment(get_source("guavy_sentiment"))
        record = _record(
            "guavy_sentiment", "BTC",
            {"date": "2026-01-05", "positive": 10, "negative": 4, "neutral": 6},
            day="2026-01-05",
        )
        row = adapter.daily_from_raw([record]).iloc[0]
        assert (row["sentiment_positive"], row["sentiment_negative"]) == (10.0, 4.0)
        assert row["sentiment_total"] == 20.0

    @pytest.mark.parametrize(
        "payload",
        [
            [{"date": "2026-01-05", "total": 1}],
            {"data": [{"date": "2026-01-05", "total": 1}]},
            {"result": {"history": [{"date": "2026-01-05", "total": 1}]}},
        ],
    )
    def test_the_wrapper_can_change_without_losing_the_capture(self, payload):
        assert guavy._series_of(payload)[0]["date"] == "2026-01-05"

    def test_counts_are_a_flow_and_get_summed(self):
        """Dos sondeos del mismo dia son dos trozos del volumen de mencion, no uno."""
        assert get_source("guavy_sentiment").aggregations()["sentiment_total"] == "sum"


# --- Wikipedia ------------------------------------------------------------------------


class TestWikipedia:
    def _record(self, language, views, day="2026-01-05", entity="BTC"):
        return _record(
            "wikipedia_pageviews", entity,
            {
                "timestamp": day.replace("-", "") + "00",
                "views": views,
                "project": f"{language}.wikipedia",
            },
            day=day, request={"language": language, "article": "Bitcoin"},
        )

    def test_concentration_separates_global_from_local_attention(self):
        adapter = wikipedia.WikipediaPageviews(get_source("wikipedia_pageviews"))
        spread = adapter.daily_from_raw(
            [self._record("en", 100), self._record("es", 100), self._record("ja", 100),
             self._record("ru", 100)]
        )
        local = adapter.daily_from_raw(
            [self._record("en", 10), self._record("es", 10), self._record("ja", 10),
             self._record("ru", 370)]
        )
        assert spread["pageviews"].iloc[0] == local["pageviews"].iloc[0] == 400.0
        assert spread["pageviews_lang_concentration"].iloc[0] == pytest.approx(0.25)
        assert local["pageviews_lang_concentration"].iloc[0] > 0.8

    def test_recapturing_the_same_window_does_not_multiply_the_views(self):
        """Cada captura re-descarga el ano: sumar seria contar las mismas visitas N veces."""
        adapter = wikipedia.WikipediaPageviews(get_source("wikipedia_pageviews"))
        frame = adapter.daily_from_raw([self._record("en", 100), self._record("en", 100)])
        assert frame["pageviews"].iloc[0] == 100.0

    def test_titles_are_a_table_because_no_rule_derives_them(self):
        adapter = wikipedia.WikipediaPageviews(get_source("wikipedia_pageviews"))
        assert adapter.article_for("SOL", "en") == "Solana (blockchain platform)"
        assert adapter.article_for("BTC", "ja") == "ビットコイン"
        assert adapter.article_for("BTC", "es") == "Bitcoin"
        assert adapter.article_for("NO_EXISTE", "en") is None


# --- P2P ------------------------------------------------------------------------------


class TestP2PPremium:
    def _record(self, prices, rate, currency="TRY", day="2026-01-05"):
        return raw_record(
            source="p2p_premium",
            entity="USDT",
            payload={
                "p2p": {"data": [{"adv": {"price": str(p)}} for p in prices]},
                "fx": {"rate": rate, "currency": currency},
            },
            day=day,
            request={"currency": currency, "trade_type": "BUY"},
        )

    def test_premium_is_measured_against_the_official_rate(self):
        adapter = p2p.BinanceP2PPremium(get_source("p2p_premium"))
        frame = adapter.daily_from_raw([self._record([52.0, 53.0, 54.0], 50.0)])
        assert frame["p2p_premium_pct"].iloc[0] == pytest.approx(6.0)  # mediana 53 vs 50

    def test_currencies_are_a_distinct_count(self):
        adapter = p2p.BinanceP2PPremium(get_source("p2p_premium"))
        frame = adapter.daily_from_raw(
            [
                self._record([52.0], 50.0, currency="TRY"),
                self._record([1600.0], 1500.0, currency="ARS"),
                self._record([53.0], 50.0, currency="TRY"),
            ]
        )
        assert frame["p2p_currencies"].iloc[0] == 2.0
        assert frame[OBSERVED].iloc[0] == 3

    def test_without_the_official_rate_there_is_no_premium(self):
        adapter = p2p.BinanceP2PPremium(get_source("p2p_premium"))
        assert adapter.daily_from_raw([self._record([52.0], None)]).empty


# --- funding --------------------------------------------------------------------------


class TestFundingDispersion:
    def _record(self, venue, rate, stamp=1_786_492_800_000):
        return raw_record(
            source="funding_dispersion",
            entity="BTC",
            payload={"fundingRate": rate, "fundingTimestamp": stamp},
            request={"venue": venue, "symbol": "BTC/USDT:USDT"},
        )

    def test_dispersion_is_between_venues_within_one_funding_window(self):
        adapter = funding_mod.FundingDispersion(get_source("funding_dispersion"))
        frame = adapter.daily_from_raw(
            [self._record("binance", 0.0001), self._record("bybit", 0.0002),
             self._record("okx", 0.0003)]
        )
        assert frame["funding_median"].iloc[0] == pytest.approx(2.0)  # 0.0002 -> 2 pb
        assert frame["funding_dispersion"].iloc[0] == pytest.approx(1.0)
        assert frame["funding_venues"].iloc[0] == 3.0

    def test_two_venues_do_not_produce_a_dispersion(self):
        adapter = funding_mod.FundingDispersion(get_source("funding_dispersion"))
        assert adapter.daily_from_raw(
            [self._record("binance", 0.0001), self._record("bybit", 0.0002)]
        ).empty

    def test_the_same_venue_archived_twice_is_still_one_venue(self):
        """Contarlo dos veces estrecharia artificialmente la dispersion."""
        adapter = funding_mod.FundingDispersion(get_source("funding_dispersion"))
        frame = adapter.daily_from_raw(
            [self._record("binance", 0.0001), self._record("binance", 0.0001),
             self._record("bybit", 0.0002), self._record("okx", 0.0003)]
        )
        assert frame["funding_venues"].iloc[0] == 3.0


# --- GitHub ---------------------------------------------------------------------------


class TestGithub:
    def test_a_week_becomes_seven_days_starting_on_its_stamp(self):
        adapter = github.GithubActivity(get_source("github_activity"))
        week_start = int(pd.Timestamp("2026-01-04", tz="UTC").timestamp())  # domingo
        record = raw_record(
            source="github_activity", entity="BTC",
            payload={"total": 63, "week": week_start, "days": [0, 10, 13, 10, 17, 12, 1]},
            day="2026-01-04", request={"repo": "bitcoin/bitcoin", "series": "commit_activity"},
        )
        frame = adapter.daily_from_raw([record])
        assert len(frame) == 7
        assert frame["commits"].sum() == 63.0
        assert frame.index.get_level_values(DAY)[0] == pd.Timestamp("2026-01-04", tz="UTC")

    def test_contributors_stay_weekly_instead_of_being_repeated(self):
        adapter = github.GithubActivity(get_source("github_activity"))
        record = raw_record(
            source="github_activity", entity="BTC",
            payload={"week_start": "2026-01-04", "contributors": 12},
            day="2026-01-04", request={"repo": "bitcoin/bitcoin", "series": "contributors"},
        )
        frame = adapter.daily_from_raw([record])
        assert len(frame) == 1 and frame["contributors"].iloc[0] == 12.0

    def test_unique_contributors_are_counted_once_per_week(self):
        authors = [
            {"weeks": [{"w": 1_767_484_800, "c": 3}, {"w": 1_768_089_600, "c": 0}]},
            {"weeks": [{"w": 1_767_484_800, "c": 1}]},
        ]
        assert github._weekly_contributors(authors) == {"2026-01-04": 2}

    def test_every_repo_of_the_universe_is_declared(self):
        from ai_trader.config import load_config
        from ai_trader.shared.entities import entity_keys

        universe = entity_keys(load_config("config/default.toml").runner.symbols)
        assert set(universe) <= set(github.REPOS)


# --- CFTC -----------------------------------------------------------------------------


class TestCftcCot:
    def test_the_day_is_when_the_report_is_published_not_when_it_is_taken(self):
        """La decision que impide meter tres dias de futuro en la serie."""
        assert cftc.published_day("2026-01-06T00:00:00.000") == "2026-01-09"
        assert cftc.PUBLICATION_LAG_DAYS == 3

    def test_nets_and_the_original_tuesday_survive_in_the_payload(self):
        adapter = cftc.CftcCot(get_source("cftc_cot"))
        payload = {
            "report_date_as_yyyy_mm_dd": "2026-01-06T00:00:00.000",
            "lev_money_positions_long": "255",
            "lev_money_positions_short": "353",
            "dealer_positions_long_all": "106",
            "dealer_positions_short_all": "0",
            "open_interest_all": "2821",
        }
        record = _record("cftc_cot", "BTC", payload, day="2026-01-09")
        frame = adapter.daily_from_raw([record])
        assert frame.index.get_level_values(DAY)[0] == pd.Timestamp("2026-01-09", tz="UTC")
        assert frame["cot_leveraged_net"].iloc[0] == -98.0
        assert frame["cot_dealer_net"].iloc[0] == 106.0
        assert record["payload"]["report_date_as_yyyy_mm_dd"].startswith("2026-01-06")

    def test_a_net_with_one_missing_leg_is_none_not_a_half_net(self):
        adapter = cftc.CftcCot(get_source("cftc_cot"))
        payload = {
            "report_date_as_yyyy_mm_dd": "2026-01-06T00:00:00.000",
            "lev_money_positions_long": "255",
            "open_interest_all": "2821",
        }
        frame = adapter.daily_from_raw([_record("cftc_cot", "BTC", payload, day="2026-01-09")])
        assert pd.isna(frame["cot_leveraged_net"].iloc[0])
        assert frame["cot_open_interest"].iloc[0] == 2821.0

    def test_only_the_two_cme_contracts_with_a_long_series(self):
        assert set(cftc.CONTRACTS) == {"BTC", "ETH"}
        assert all("CHICAGO MERCANTILE" in name for name in cftc.CONTRACTS.values())


# --- normalizacion ---------------------------------------------------------------------


def _series_frame(entity: str, values, start="2026-01-01") -> pd.DataFrame:
    days = pd.date_range(start, periods=len(values), tz="UTC")
    frame = pd.DataFrame(
        {ENTITY: entity, DAY: days, "flow": values, OBSERVED: 1}
    ).set_index([ENTITY, DAY])
    return frame


class TestNormalization:
    def test_the_self_z_is_causal(self):
        """Un pico al final NO puede cambiar la z de los dias anteriores. Es la forma mas
        silenciosa de meter futuro en un backtest: no rompe nada y mejora el resultado."""
        base = list(np.linspace(1.0, 2.0, 60))
        quiet = normalize_features(_series_frame("BTC", base))
        with_spike = normalize_features(_series_frame("BTC", [*base[:-1], 1000.0]))
        pd.testing.assert_series_equal(
            quiet["flow_z"].iloc[:-1], with_spike["flow_z"].iloc[:-1]
        )

    def test_the_declared_clip_holds(self):
        values = [*np.linspace(1.0, 2.0, 60), 1e9]
        frame = normalize_features(_series_frame("BTC", values))
        assert frame["flow_z"].abs().max() <= Z_CLIP
        assert normalization_spec()["clip"] == Z_CLIP

    def test_a_young_listing_has_no_self_z_but_does_have_a_cross_section_one(self):
        """La propiedad que hace que esto sirva igual a BTC que a un listado de ayer."""
        frames = [
            _series_frame(f"OLD{i}", list(np.linspace(1.0, 2.0, 40) * (i + 1))) for i in range(5)
        ]
        newcomer = _series_frame("NEW", [5.0, 6.0], start="2026-01-01")
        joined = pd.concat([*frames, newcomer]).sort_index()

        out = normalize_features(joined)
        new_rows = out.xs("NEW", level=ENTITY)
        assert new_rows["flow_z"].isna().all()  # sin historia propia
        assert new_rows["flow_x"].notna().all()  # pero si comparada con el resto de hoy

    def test_a_thin_day_produces_no_cross_section_z(self):
        out = normalize_features(_series_frame("BTC", [1.0, 2.0, 3.0]))
        assert out["flow_x"].isna().all()
        assert normalization_spec()["min_cross_section"] == MIN_CROSS_SECTION

    def test_short_history_produces_no_self_z(self):
        out = normalize_features(_series_frame("BTC", list(range(MIN_HISTORY - 1))))
        assert out["flow_z"].isna().all()

    def test_a_constant_series_has_no_scale_and_is_not_infinite(self):
        out = normalize_features(_series_frame("BTC", [7.0] * 40))
        assert out["flow_z"].isna().all()

    def test_gaps_are_nan_and_never_zero(self):
        """0.0 significaria 'normal, en la media', que no se ha observado."""
        out = normalize_features(_series_frame("BTC", [1.0, 2.0]))
        assert out["flow_z"].isna().all()
        assert normalization_spec()["missing"].startswith("NaN")

    def test_raw_columns_survive_next_to_the_normalized_ones(self):
        out = normalize_features(_series_frame("BTC", [1.0, 2.0]))
        assert list(out.columns) == ["flow", f"flow{SUFFIX_SELF}", f"flow{SUFFIX_CROSS}", OBSERVED]

    def test_the_panel_publishes_only_normalized_columns(self):
        frames = {
            "uno": _series_frame("BTC", list(np.linspace(1, 2, 30))),
            "dos": _series_frame("BTC", list(np.linspace(5, 6, 30))).rename(
                columns={"flow": "level"}
            ),
        }
        wide = panel(frames)
        assert "flow" not in wide.columns and "level" not in wide.columns
        assert {"flow_z", "flow_x", "level_z", "level_x", OBSERVED} == set(wide.columns)
        assert wide[OBSERVED].iloc[0] == 2  # una observacion por fuente

    def test_coverage_separates_missing_calendar_from_missing_universe(self):
        wide = panel({"uno": _series_frame("BTC", list(np.linspace(1, 2, 30)))})
        coverage = normalization_coverage(wide)
        assert coverage["rows"] == 30
        assert coverage["cross_pct"] == 0.0  # una sola entidad: no hay seccion cruzada
        assert coverage["self_pct"] > 0.0


# --- profundidad MEDIDA -----------------------------------------------------------------


class TestDepth:
    def test_measuring_from_the_archive_needs_no_network(self, tmp_path):
        store = SignalStore(tmp_path)
        source = get_source("defillama_stablecoins")
        records = [
            _record(
                source.key, "ethereum",
                {"date": str(int(pd.Timestamp(day).timestamp())),
                 "totalCirculatingUSD": {"peggedUSD": 100.0}},
                day=day,
            )
            for day in ("2025-11-20", "2026-01-05")
        ]
        store.append(source.key, records)

        report = measure_depth(sources=[source], store=store, from_archive=True, ledger_path=None)
        measured = report.sources[0]
        assert measured.method == "archive"
        assert measured.first_day == "2025-11-20"
        assert measured.entities["ethereum"] == "2025-11-20"

    def test_a_source_without_credential_declares_its_error_instead_of_a_hole(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        report = measure_depth(
            sources=[get_source("fred_macro")], store=SignalStore(tmp_path), ledger_path=None
        )
        row = report.sources[0]
        assert row.first_day is None
        assert "FRED_API_KEY" in (row.error or "")

    def test_the_ledger_roundtrips(self, tmp_path):
        path = tmp_path / "depth.json"
        measure_depth(
            sources=[get_source("cftc_cot")],
            store=SignalStore(tmp_path / "raw"),
            from_archive=True,
            ledger_path=path,
        )
        assert load_ledger(path)["n_sources"] == 1
        assert load_ledger(tmp_path / "no_existe.json") is None

    def test_declared_history_is_backed_by_a_measurement(self):
        """LA regla del catalogo, hecha mecanica: nadie puede escribir un `history_from`
        que el registro de mediciones no respalde. Sin este test, la unica defensa contra
        copiar la fecha del folleto comercial seria la disciplina."""
        ledger = measured_history() if DEPTH_LEDGER.exists() else {}
        for source in CATALOG:
            if source.history_from is None:
                continue
            assert source.key in ledger, (
                f"'{source.key}' declara history_from sin medicion en {DEPTH_LEDGER}"
            )
            assert source.history_from == ledger[source.key], (
                f"'{source.key}' declara {source.history_from} y lo medido es "
                f"{ledger[source.key]}"
            )

    def test_one_measured_day_does_not_earn_a_history_from(self, tmp_path):
        """Una fuente `forward_capture` tiene su primer dia el dia que arranco la captura.
        Declararlo seria literalmente cierto y practicamente enganoso: `backtestable`
        pasaria a True con una serie de un dia."""
        path = tmp_path / "depth.json"
        path.write_text(
            json.dumps(
                {
                    "sources": [
                        {"source_key": "p2p_premium", "first_day": "2026-08-11", "days": 1},
                        {"source_key": "cftc_cot", "first_day": "2018-04-13", "days": 714},
                    ]
                }
            ),
            encoding="utf-8",
        )
        declarable = measured_history(path)
        assert "p2p_premium" not in declarable
        assert declarable["cftc_cot"] == date(2018, 4, 13)
        assert "p2p_premium" in measured_history(path, min_days=0)  # medido, pero no declarable
        assert get_source("p2p_premium").history_from is None

    def test_the_comparison_reports_both_directions(self, tmp_path):
        path = tmp_path / "depth.json"
        path.write_text(
            json.dumps(
                {
                    "sources": [
                        {"source_key": "cftc_cot", "first_day": "2017-12-22"},
                        {"source_key": "etf_flows", "first_day": None},
                    ]
                }
            ),
            encoding="utf-8",
        )
        comparison = declared_vs_measured(path)
        assert comparison["cftc_cot"]["measured"] == "2017-12-22"
        assert comparison["etf_flows"]["measured"] is None

    def test_source_depth_serializes(self):
        depth = SourceDepth(
            source_key="x", measured_at=datetime(2026, 1, 5, tzinfo=UTC).isoformat(),
            method="provider", connected=True, first_day="2024-01-11", last_day="2026-01-05",
            days=10, observations=12, entities={"BTC": "2024-01-11"},
        )
        assert depth.measured and depth.as_dict()["n_entities"] == 1
        assert date.fromisoformat(depth.as_dict()["first_day"]).year == 2024
