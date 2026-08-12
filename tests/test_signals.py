"""
Tests del esqueleto de senales: esquema, catalogo, puerto, archivo, captura y auditoria.

Ninguno toca la red. El puerto esta partido en dos capas justamente para que la mitad que
se equivoca —el mapeo del payload— sea testeable sin proveedor, y estos tests son la
comprobacion de que esa promesa se cumple.
"""
from __future__ import annotations

import gzip
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from ai_trader.data.providers.ccxt_crypto import CCXTCrypto
from ai_trader.observation import regime
from ai_trader.shared.bars import normalize_bars
from ai_trader.shared.clock import visible_cutoff
from ai_trader.shared.entities import (
    ENTITY_OVERRIDES,
    MARKET_ENTITY,
    EntityKind,
    entity_keys,
    resolve_entity,
)
from ai_trader.shared.instruments import QUOTE_CURRENCIES, split_pair
from ai_trader.shared.signals import (
    AGG_LAST,
    AGG_SUM,
    DAY,
    ENTITY,
    OBSERVED,
    coverage,
    empty_signals,
    merge_signals,
    normalize_signals,
    visible_signals,
)
from ai_trader.signals import audit as audit_module
from ai_trader.signals import capture as capture_module
from ai_trader.signals import source as source_module
from ai_trader.signals.audit import audit_archive, audit_entities
from ai_trader.signals.capture import capture, entities_for
from ai_trader.signals.catalog import (
    CATALOG,
    PIT_ARCHIVE_REVISABLE,
    PIT_FORWARD_CAPTURE,
    SCOPE_ASSET,
    SCOPE_MARKET,
    TIER_MECHANICAL,
    TIER_STATISTICAL,
    SignalFeature,
    SignalSource,
    backtestable_sources,
    catalog_summary,
    get_source,
    sources_by_tier,
)
from ai_trader.signals.source import (
    RAW_FETCHED_AT,
    BaseJsonAdapter,
    build_adapter,
    raw_record,
    register_adapter,
)
from ai_trader.signals.store import DerivedCache, SignalStore, month_of, safe_name

UTC = timezone.utc


# --- deuda pagada: visible_cutoff y la tupla de quotes ------------------------------


class TestSharedConventions:
    def test_visible_cutoff_excludes_today(self):
        now = datetime(2026, 3, 15, 23, 59, tzinfo=UTC)
        assert visible_cutoff(now) == pd.Timestamp("2026-03-15", tz="UTC")

    def test_visible_cutoff_assumes_utc_when_naive(self):
        assert visible_cutoff(datetime(2026, 3, 15, 10)) == pd.Timestamp("2026-03-15", tz="UTC")

    def test_regime_imports_the_shared_cutoff(self):
        """La tercera copia no se escribio: el regimen usa LA funcion, no una igual."""
        assert regime.visible_cutoff is visible_cutoff

    def test_quotes_are_ordered_longest_first(self):
        """'XUSDT' no puede partirse como 'XUSD' + 'T'."""
        assert QUOTE_CURRENCIES.index("USDT") < QUOTE_CURRENCIES.index("USD")
        assert split_pair("PEPEUSDT") == ("PEPE", "USDT")

    @pytest.mark.parametrize(
        "symbol,expected",
        [
            ("BTC/USDT", ("BTC", "USDT")),
            ("btc-usdt", ("BTC", "USDT")),
            ("ETHUSD", ("ETH", "USD")),
            ("BTC/FOO", ("BTC", "FOO")),  # con separador no se exige quote conocida
            ("AAPL", None),
            ("/USDT", None),
            ("PM::whatever", None),
            ("", None),
        ],
    )
    def test_split_pair(self, symbol, expected):
        assert split_pair(symbol) == expected

    def test_ccxt_normalizes_through_the_shared_rule(self):
        provider = CCXTCrypto()  # perezoso: no abre exchange
        assert provider.normalize_symbol("btc-usdt") == "BTC/USDT"
        assert provider.normalize_symbol("ETHUSD") == "ETH/USD"
        with pytest.raises(ValueError):
            provider.normalize_symbol("AAPL")


# --- entidades ----------------------------------------------------------------------


class TestEntities:
    def test_override_table_starts_empty(self):
        """Invariante del diseno: la tabla crece con fallos OBSERVADOS, no antes."""
        assert ENTITY_OVERRIDES == {}

    @pytest.mark.parametrize(
        "symbol,key,kind",
        [
            ("BTC/USDT", "BTC", EntityKind.TOKEN),
            ("sol/usdt", "SOL", EntityKind.TOKEN),
            ("ETHUSD", "ETH", EntityKind.TOKEN),
            ("SPY", "SPY", EntityKind.EQUITY),
            ("PM::us-election-2028", "US-ELECTION-2028", EntityKind.PREDICTION),
        ],
    )
    def test_rule_resolves_without_any_table(self, symbol, key, kind):
        ref = resolve_entity(symbol)
        assert (ref.key, ref.kind, ref.source) == (key, kind, "rule")
        assert ref.resolved

    def test_a_new_listing_resolves_the_day_it_appears(self):
        """La propiedad que justifica la regla derivada frente a una tabla curada."""
        ref = resolve_entity("BRANDNEW/USDT")
        assert ref.key == "BRANDNEW" and ref.source == "rule"

    @pytest.mark.parametrize("symbol", ["", "   ", "/USDT", "PM::", "BTC USDT!"])
    def test_unresolvable_symbols_are_declared_not_guessed(self, symbol):
        ref = resolve_entity(symbol)
        assert ref.source == "unmapped"
        assert not ref.resolved
        assert ref.key == ""

    def test_override_wins_over_the_rule(self, monkeypatch):
        monkeypatch.setitem(ENTITY_OVERRIDES, "1000PEPE/USDT", "PEPE")
        ref = resolve_entity("1000pepe/usdt")
        assert ref.key == "PEPE"
        assert ref.source == "override"
        assert resolve_entity("PEPE/USDT").source == "rule"

    def test_entity_keys_deduplicate_the_same_asset(self):
        assert entity_keys(["BTC/USDT", "BTC/USD", "ETH/USDT"]) == ("BTC", "ETH")

    def test_bare_ticker_is_equity_by_the_canonical_rule(self):
        """Limite declarado en el modulo: 'BTC' sin par es renta variable, y para eso
        existe la tabla de overrides el dia que se observe."""
        assert resolve_entity("BTC").kind is EntityKind.EQUITY


# --- esquema canonico ---------------------------------------------------------------


def _raw_rows():
    """Tres observaciones del MISMO dia y entidad, mas otra entidad y otro dia."""
    return pd.DataFrame(
        [
            {ENTITY: "BTC", DAY: "2026-01-05T02:00:00Z", "flow": 10.0, "level": 1.0},
            {ENTITY: "BTC", DAY: "2026-01-05T10:00:00Z", "flow": 20.0, "level": 2.0},
            {ENTITY: "BTC", DAY: "2026-01-05T23:30:00Z", "flow": 30.0, "level": 3.0},
            {ENTITY: "BTC", DAY: "2026-01-06", "flow": 5.0, "level": 9.0},
            {ENTITY: "ETH", DAY: "2026-01-05", "flow": 1.0, "level": 7.0},
        ]
    )


class TestSignalSchema:
    def test_several_observations_of_a_day_are_aggregated_not_dropped(self):
        frame = normalize_signals(
            _raw_rows(), features=["flow", "level"], agg={"flow": AGG_SUM, "level": AGG_LAST}
        )
        row = frame.loc[("BTC", pd.Timestamp("2026-01-05", tz="UTC"))]
        assert row["flow"] == 60.0  # 10+20+30: NINGUNA observacion se perdio
        assert row["level"] == 3.0  # ultimo estado del dia
        assert row[OBSERVED] == 3

    def test_normalize_bars_would_have_destroyed_them(self):
        """El motivo explicito de que este esquema no reuse `normalize_bars`.

        El caso: tres emisores publicando su flujo del MISMO dia. Con `keep='last'` el
        total del dia serian 30 en vez de 60, y sin ninguna senal de que faltan dos filas.
        """
        by_issuer = pd.DataFrame(
            [
                {DAY: "2026-01-05", ENTITY: "BTC", "flow": 10.0},
                {DAY: "2026-01-05", ENTITY: "BTC", "flow": 20.0},
                {DAY: "2026-01-05", ENTITY: "BTC", "flow": 30.0},
            ]
        )
        destroyed = normalize_bars(by_issuer.set_index(DAY))
        assert len(destroyed) == 1 and destroyed["flow"].sum() == 30.0

        kept = normalize_signals(by_issuer, features=["flow"], agg=AGG_SUM)
        assert len(kept) == 1
        assert kept["flow"].iloc[0] == 60.0 and kept[OBSERVED].iloc[0] == 3

    def test_index_is_entity_and_day_sorted_and_unique(self):
        frame = normalize_signals(_raw_rows(), features=["flow", "level"])
        assert list(frame.index.names) == [ENTITY, DAY]
        assert frame.index.is_unique
        assert list(frame.index) == sorted(frame.index)
        assert len(frame) == 3  # (BTC,5) (BTC,6) (ETH,5)

    def test_two_entities_of_the_same_day_both_survive(self):
        frame = normalize_signals(_raw_rows(), features=["flow"])
        day = pd.Timestamp("2026-01-05", tz="UTC")
        assert ("BTC", day) in frame.index and ("ETH", day) in frame.index

    def test_time_of_day_is_dropped(self):
        frame = normalize_signals(_raw_rows(), features=["flow"])
        assert all(ts == ts.normalize() for ts in frame.index.get_level_values(DAY))

    def test_rows_without_entity_or_day_are_discarded(self):
        raw = pd.DataFrame(
            [
                {ENTITY: "BTC", DAY: "2026-01-05", "flow": 1.0},
                {ENTITY: "", DAY: "2026-01-05", "flow": 2.0},
                {ENTITY: "BTC", DAY: "no-es-una-fecha", "flow": 3.0},
            ]
        )
        frame = normalize_signals(raw, features=["flow"])
        assert len(frame) == 1 and frame[OBSERVED].iloc[0] == 1

    def test_observed_survives_reaggregation(self):
        once = normalize_signals(_raw_rows(), features=["flow"], agg=AGG_SUM)
        twice = normalize_signals(once, features=["flow"], agg=AGG_SUM)
        pd.testing.assert_frame_equal(once, twice)

    def test_merge_is_associative_on_observed(self):
        first = normalize_signals(_raw_rows().head(2), features=["flow"], agg=AGG_SUM)
        second = normalize_signals(_raw_rows().tail(3), features=["flow"], agg=AGG_SUM)
        merged = merge_signals([first, second], features=["flow"], agg=AGG_SUM)
        assert int(merged[OBSERVED].sum()) == 5

    def test_empty_frame_is_canonical(self):
        frame = empty_signals(["flow"])
        assert list(frame.columns) == ["flow", OBSERVED]
        assert frame.empty and list(frame.index.names) == [ENTITY, DAY]

    def test_visible_signals_excludes_today(self):
        frame = normalize_signals(_raw_rows(), features=["flow"])
        visible = visible_signals(frame, datetime(2026, 1, 6, 15, tzinfo=UTC))
        assert set(visible.index.get_level_values(DAY)) == {pd.Timestamp("2026-01-05", tz="UTC")}

    def test_unknown_aggregation_is_rejected(self):
        with pytest.raises(ValueError, match="Agregacion desconocida"):
            normalize_signals(_raw_rows(), features=["flow"], agg="median_ponderada")

    def test_missing_columns_are_reported(self):
        with pytest.raises(KeyError):
            normalize_signals(_raw_rows(), features=["no_existe"])
        with pytest.raises(KeyError):
            normalize_signals(pd.DataFrame([{"foo": 1}]))

    def test_coverage_summary(self):
        frame = normalize_signals(_raw_rows(), features=["flow"], agg=AGG_SUM)
        summary = coverage(frame)
        assert summary["BTC"] == {
            "days": 2,
            "observations": 4,
            "first_day": "2026-01-05",
            "last_day": "2026-01-06",
        }


# --- catalogo -----------------------------------------------------------------------


class TestCatalog:
    def test_every_declared_source_is_valid(self):
        assert len(CATALOG) == len({s.key for s in CATALOG})
        for source in CATALOG:
            assert source.features and source.license
            assert source.aggregations().keys() == set(source.feature_names)
            assert safe_name(source.key) == source.key  # sirve de nombre de directorio

    def test_backtestable_is_exactly_what_was_measured(self):
        """El dia cero esto decia "ninguna". Cambio cuando alguien midio, que es justo
        cuando debia cambiar: hoy son backtesteables las fuentes que la sonda
        (`signals/depth.py`) pudo medir, y solo esas. Las que siguen a None no es que no
        tengan historia: es que nadie la ha comprobado todavia."""
        measured = backtestable_sources()
        assert catalog_summary()["n_backtestable"] == len(measured)
        assert all(s.history_from is not None for s in measured)
        assert {s.key for s in CATALOG if s.history_from} == {s.key for s in measured}

    def test_backtestable_follows_measured_history(self):
        measured = SignalSource(
            key="medida",
            title="t",
            tier=TIER_STATISTICAL,
            scope=SCOPE_ASSET,
            cadence="daily",
            entity_kind=EntityKind.TOKEN,
            features=(SignalFeature("x"),),
            pit=PIT_ARCHIVE_REVISABLE,
            license="abierta",
            history_from=date(2024, 1, 1),
        )
        assert measured.backtestable
        assert measured.as_dict()["history_from"] == "2024-01-01"

    def test_both_doors_are_populated(self):
        assert sources_by_tier(TIER_MECHANICAL) and sources_by_tier(TIER_STATISTICAL)
        assert all(s.tier == TIER_MECHANICAL for s in sources_by_tier(TIER_MECHANICAL))

    def test_market_scoped_sources_have_one_entity(self):
        for source in CATALOG:
            if source.scope == SCOPE_MARKET:
                assert source.is_market_scoped

    def test_unknown_key_lists_the_catalog(self):
        with pytest.raises(KeyError, match="Catalogo"):
            get_source("no_existe")

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"tier": "C"}, "Tier desconocido"),
            ({"scope": "galaxia"}, "Scope desconocido"),
            ({"cadence": "cada_luna"}, "Cadencia desconocida"),
            ({"pit": "confia_en_mi"}, "pit desconocido"),
            ({"license": ""}, "licencia"),
            ({"features": ()}, "ninguna feature"),
            ({"key": "Mayusculas"}, "Clave de fuente invalida"),
            ({"entity_kind": EntityKind.CHAIN}, "no encaja con scope"),
        ],
    )
    def test_declarations_are_validated(self, kwargs, match):
        base = {
            "key": "prueba",
            "title": "t",
            "tier": TIER_STATISTICAL,
            "scope": SCOPE_ASSET,
            "cadence": "daily",
            "entity_kind": EntityKind.TOKEN,
            "features": (SignalFeature("x"),),
            "pit": PIT_FORWARD_CAPTURE,
            "license": "abierta",
        }
        with pytest.raises(ValueError, match=match):
            SignalSource(**{**base, **kwargs})

    def test_repeated_features_within_a_source_are_rejected(self):
        with pytest.raises(ValueError, match="Features repetidas"):
            SignalSource(
                key="prueba",
                title="t",
                tier=TIER_STATISTICAL,
                scope=SCOPE_ASSET,
                cadence="daily",
                entity_kind=EntityKind.TOKEN,
                features=(SignalFeature("x"), SignalFeature("x")),
                pit=PIT_FORWARD_CAPTURE,
                license="abierta",
            )

    def test_feature_names_are_unique_across_the_whole_catalog(self):
        """El dia que estas columnas se junten en un vector, dos homonimas se pisarian."""
        names = [name for source in CATALOG for name in source.feature_names]
        assert len(names) == len(set(names))

    def test_no_secret_lives_in_the_catalog(self):
        for source in CATALOG:
            assert source.auth_env is None or source.auth_env.isupper()


# --- puerto (dos capas) -------------------------------------------------------------

TEST_SOURCE = SignalSource(
    key="fuente_de_prueba",
    title="Fuente de prueba",
    tier=TIER_STATISTICAL,
    scope=SCOPE_ASSET,
    cadence="daily",
    entity_kind=EntityKind.TOKEN,
    features=(SignalFeature("flow", AGG_SUM), SignalFeature("level", AGG_LAST)),
    pit=PIT_FORWARD_CAPTURE,
    license="prueba",
)

EVENT_SOURCE = SignalSource(
    key="fuente_por_evento",
    title="Fuente por evento",
    tier=TIER_MECHANICAL,
    scope=SCOPE_MARKET,
    cadence="event",
    entity_kind=EntityKind.MARKET,
    features=(SignalFeature("hits", AGG_SUM),),
    pit=PIT_FORWARD_CAPTURE,
    license="prueba",
)


class FakeAdapter(BaseJsonAdapter):
    """Adaptador de juguete: la capa 1 devuelve un payload enlatado (no hay red) y la
    capa 2 lo traduce. Es la forma exacta que tendra un adaptador real."""

    PAYLOADS = {
        "BTC": [
            {"t": "2026-01-05T01:00:00Z", "f": 10, "l": 1},
            {"t": "2026-01-05T13:00:00Z", "f": 20, "l": 2},
            {"t": "2026-01-06T01:00:00Z", "f": 5, "l": 9},
        ],
        "ETH": [{"t": "2026-01-05T01:00:00Z", "f": 1, "l": 7}],
    }

    def fetch_raw(self, entities, start=None, end=None):
        out = []
        for entity in entities:
            for item in self.PAYLOADS.get(entity, []):
                out.append(self.record(entity, item, day=item["t"][:10]))
        return out

    def daily_from_raw(self, records):
        rows = [
            {
                ENTITY: r["entity"],
                DAY: r["payload"]["t"],
                "flow": r["payload"]["f"],
                "level": r["payload"]["l"],
            }
            for r in records
        ]
        return self.to_daily(rows)


class TestPort:
    def test_the_registry_holds_the_whole_catalog(self):
        """El dia cero esto decia "vacio"; despues, "solo las continuas"; hoy, "todas". Es
        la misma pregunta cada vez —que fuentes existen de verdad— y la respuesta ya no
        depende del tier: las de evento tambien tienen adaptador, porque sin adaptador no
        hay medicion y sin medicion no hay tamano de muestra que discutir."""
        from ai_trader.signals.capture import connect_adapters

        connected = set(connect_adapters())
        assert connected == {s.key for s in CATALOG}
        assert all(build_adapter(s) is not None for s in CATALOG if s.tier == TIER_MECHANICAL)
        assert all(build_adapter(s) is not None for s in CATALOG if s.tier == TIER_STATISTICAL)

    def test_raw_record_stamps_fetched_at(self):
        record = raw_record(source="s", entity="BTC", payload={"a": 1})
        assert record[RAW_FETCHED_AT]
        assert record["payload"] == {"a": 1}
        with pytest.raises(ValueError):
            raw_record(source="", entity="BTC", payload={})

    def test_derivation_is_pure_and_needs_no_network(self):
        adapter = FakeAdapter(TEST_SOURCE)
        records = adapter.fetch_raw(["BTC", "ETH"])
        frame = adapter.daily_from_raw(records)

        row = frame.loc[("BTC", pd.Timestamp("2026-01-05", tz="UTC"))]
        assert row["flow"] == 30.0  # politica del CATALOGO: suma
        assert row["level"] == 2.0  # politica del CATALOGO: ultimo
        assert row[OBSERVED] == 2
        assert list(frame.columns) == ["flow", "level", OBSERVED]

    def test_rederiving_from_the_archive_gives_the_same_frame(self, tmp_path):
        """La propiedad que justifica las dos capas: corregir un mapeo NO re-descarga."""
        adapter = FakeAdapter(TEST_SOURCE)
        store = SignalStore(tmp_path)
        records = adapter.fetch_raw(["BTC", "ETH"])
        store.append(TEST_SOURCE.key, records)

        from_memory = adapter.daily_from_raw(records)
        from_disk = adapter.daily_from_raw(store.read(TEST_SOURCE.key))
        pd.testing.assert_frame_equal(from_memory, from_disk)

    def test_http_client_is_lazy(self):
        adapter = BaseJsonAdapter(TEST_SOURCE)  # sin endpoint declarado
        with pytest.raises(RuntimeError, match="endpoint"):
            _ = adapter.client

    def test_register_adapter_guards_the_catalog(self, monkeypatch):
        monkeypatch.setattr(source_module, "REGISTRY", {})
        with pytest.raises(KeyError):
            register_adapter("fuente_inventada", FakeAdapter)
        register_adapter("etf_flows", FakeAdapter)
        with pytest.raises(ValueError, match="Ya hay un adaptador"):
            register_adapter("etf_flows", FakeAdapter)


# --- archivo crudo ------------------------------------------------------------------


class TestStore:
    def test_paths_are_safe_on_both_filesystems(self):
        assert safe_name("BTC/USDT") == "BTC_USDT"
        assert safe_name(MARKET_ENTITY) == "_market"
        assert safe_name("") == "_unknown"

    def test_roundtrip(self, tmp_path):
        store = SignalStore(tmp_path)
        records = FakeAdapter(TEST_SOURCE).fetch_raw(["BTC"])
        assert store.append(TEST_SOURCE.key, records) == 3
        assert len(store.read(TEST_SOURCE.key, "BTC")) == 3
        assert store.entities(TEST_SOURCE.key) == ("BTC",)
        assert store.sources() == (TEST_SOURCE.key,)

    def test_append_only_keeps_the_revision_and_the_original(self, tmp_path):
        """Una fuente `archive_revisable` que reescribe el pasado deja DOS lineas."""
        store = SignalStore(tmp_path)
        original = raw_record(source="s", entity="BTC", payload={"v": 1}, day="2026-01-05")
        revised = raw_record(source="s", entity="BTC", payload={"v": 2}, day="2026-01-05")
        store.append("s", [original])
        store.append("s", [revised])

        lines = store.read("s", "BTC")
        assert [line["payload"]["v"] for line in lines] == [1, 2]

    def test_shards_by_observation_month_not_by_download_day(self, tmp_path):
        store = SignalStore(tmp_path)
        backfill = [
            raw_record(source="s", entity="BTC", payload={}, day=f"2025-0{m}-15") for m in (1, 2, 3)
        ]
        store.append("s", backfill)
        assert store.months("s", "BTC") == ("2025-01", "2025-02", "2025-03")

    def test_falls_back_to_fetched_at_when_there_is_no_observation_day(self):
        record = raw_record(
            source="s", entity="BTC", payload={}, fetched_at=datetime(2026, 5, 2, tzinfo=UTC)
        )
        assert month_of(record) == "2026-05"

    def test_a_record_without_fetched_at_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="fetched_at"):
            SignalStore(tmp_path).append("s", [{"entity": "BTC", "payload": {}}])

    def test_unreadable_lines_do_not_lose_the_month(self, tmp_path):
        store = SignalStore(tmp_path)
        store.append("s", [raw_record(source="s", entity="BTC", payload={"v": 1}, day="2026-01-05")])
        path = store.shard_path("s", "BTC", "2026-01")
        with gzip.open(path, "at", encoding="utf-8") as handle:
            handle.write("{esto no es json\n")

        assert len(store.read("s", "BTC")) == 1

    def test_stats(self, tmp_path):
        store = SignalStore(tmp_path)
        store.append(TEST_SOURCE.key, FakeAdapter(TEST_SOURCE).fetch_raw(["BTC", "ETH"]))
        stats = store.stats()[TEST_SOURCE.key]
        assert stats == {"entities": 2, "months": ["2026-01"], "records": 4}


class TestDerivedCache:
    def test_roundtrip_and_clear(self, tmp_path):
        cache = DerivedCache(tmp_path)
        frame = FakeAdapter(TEST_SOURCE).daily_from_raw(
            FakeAdapter(TEST_SOURCE).fetch_raw(["BTC"])
        )
        cache.save(TEST_SOURCE.key, "BTC", frame)

        loaded = cache.load(TEST_SOURCE.key, "BTC")
        pd.testing.assert_frame_equal(loaded, frame, check_dtype=False)

        assert cache.clear(TEST_SOURCE.key) == 1
        assert cache.load(TEST_SOURCE.key, "BTC") is None

    def test_missing_entry_is_none_not_an_error(self, tmp_path):
        assert DerivedCache(tmp_path).load("lo_que_sea", "BTC") is None


# --- captura ------------------------------------------------------------------------


class TestCapture:
    def test_entities_come_from_the_universe_filtered_by_kind(self):
        universe = ["BTC/USDT", "ETH/USDT", "SPY", "PM::algo"]
        assert entities_for(TEST_SOURCE, universe) == ("BTC", "ETH")

    def test_market_scoped_sources_ask_for_one_entity(self):
        assert entities_for(EVENT_SOURCE, ["BTC/USDT"]) == (MARKET_ENTITY,)

    def test_declared_entities_win_over_the_universe(self):
        chains = get_source("defillama_stablecoins")
        assert entities_for(chains, ["BTC/USDT"]) == chains.fixed_entities

    def test_a_pass_over_the_real_catalog_publishes_what_falta(self, tmp_path, monkeypatch):
        """La pasada completa, sin red: con el registro vaciado, TODAS las fuentes salen
        como pendientes. Es el mismo informe del dia cero, y lo que mide ahora es cuantas
        siguen sin adaptador —las seis mecanicas— y no cuantas hay declaradas."""
        monkeypatch.setattr(source_module, "REGISTRY", {})
        monkeypatch.setattr(capture_module, "connect_adapters", lambda: ())
        report = capture(
            ["BTC/USDT", "ETH/USDT"],
            store=SignalStore(tmp_path / "raw"),
            report_path=tmp_path / "capture.json",
        )
        assert report.n_sources == len(CATALOG)
        assert report.n_connected == 0
        assert report.n_pending == len(CATALOG)
        assert report.records == 0
        assert (tmp_path / "capture.json").exists()
        assert capture_module.load_capture_report(tmp_path / "capture.json")["n_pending"] == len(
            CATALOG
        )

    def test_a_capture_connects_the_batch_by_itself(self, tmp_path):
        """La captura NO puede depender de que alguien se acuerde de registrar: un registro
        vacio archivaria cero y el informe diria 'todo correcto, 0 registros'."""
        monkeypatched_registry = dict(source_module.REGISTRY)
        source_module.REGISTRY.clear()
        try:
            report = capture([], store=SignalStore(tmp_path), sources=[], report_path=None)
            assert report.n_sources == 0
            assert set(source_module.REGISTRY) >= {"etf_flows", "cftc_cot"}
        finally:
            source_module.REGISTRY.update(monkeypatched_registry)

    def test_archives_when_an_adapter_exists(self, tmp_path, monkeypatch):
        monkeypatch.setitem(source_module.REGISTRY, TEST_SOURCE.key, FakeAdapter)
        store = SignalStore(tmp_path)
        report = capture(
            ["BTC/USDT", "ETH/USDT"], store=store, sources=[TEST_SOURCE], report_path=None
        )
        assert report.n_connected == 1
        assert report.records == 4
        assert len(store.read(TEST_SOURCE.key)) == 4

    def test_one_broken_source_does_not_stop_the_others(self, tmp_path, monkeypatch):
        class Broken(FakeAdapter):
            def fetch_raw(self, entities, start=None, end=None):
                raise RuntimeError("502 del proveedor")

        monkeypatch.setitem(source_module.REGISTRY, TEST_SOURCE.key, FakeAdapter)
        monkeypatch.setitem(source_module.REGISTRY, EVENT_SOURCE.key, Broken)
        report = capture(
            ["BTC/USDT", "ETH/USDT"],
            store=SignalStore(tmp_path),
            sources=[EVENT_SOURCE, TEST_SOURCE],
            report_path=None,
        )
        assert report.n_failed == 1
        assert report.records == 4  # la fuente sana capturo igual
        errors = [s.error for s in report.sources if s.error]
        assert "502" in errors[0]

    def test_report_is_serializable(self, tmp_path):
        report = capture([], store=SignalStore(tmp_path), sources=[TEST_SOURCE], report_path=None)
        data = report.as_dict()
        assert data["catalog"]["n_sources"] == len(CATALOG)
        assert data["sources"][0]["connected"] is False


# --- auditoria ----------------------------------------------------------------------


class TestEntityAudit:
    def test_the_configured_universe_resolves_by_rule(self):
        from ai_trader.config import load_config

        universe = load_config("config/default.toml").runner.symbols
        result = audit_entities(universe)
        assert result.n_symbols == len(universe)
        assert result.coverage_pct == 100.0
        assert result.by_source["override"] == 0
        assert result.unmapped == ()

    def test_collisions_are_published_not_hidden(self):
        result = audit_entities(["BTC/USDT", "BTC/USD", "ETH/USDT"])
        assert result.collisions == {"BTC": ("BTC/USDT", "BTC/USD")}

    def test_unmapped_symbols_are_the_only_candidates_to_the_override_table(self):
        result = audit_entities(["BTC/USDT", "?!"])
        assert result.unmapped == ("?!",)
        assert result.by_source["unmapped"] == 1
        assert result.as_dict()["coverage_pct"] == 50.0


class TestArchiveAudit:
    def _archive(self, tmp_path, days, entity="BTC", source=TEST_SOURCE):
        store = SignalStore(tmp_path)
        store.append(
            source.key,
            [raw_record(source=source.key, entity=entity, payload={}, day=d) for d in days],
        )
        return store

    def test_empty_archive_measures_zero_without_failing(self, tmp_path):
        result = audit_archive(SignalStore(tmp_path / "vacio"))
        assert result.records == 0
        assert result.n_with_archive == 0
        assert len(result.sources) == len(CATALOG)
        assert all(s.measured_history_from is None for s in result.sources)

    def test_coverage_is_measured_over_the_span_actually_captured(self, tmp_path):
        days = [f"2026-01-{d:02d}" for d in range(1, 11)]
        store = self._archive(tmp_path, days)
        result = audit_archive(store, sources=[TEST_SOURCE])

        year = result.sources[0].years[0]
        assert (year.entity, year.year, year.days) == ("BTC", 2026, 10)
        assert year.first_day == "2026-01-01" and year.last_day == "2026-01-10"
        assert year.coverage_pct == 100.0
        assert year.max_gap_days == 1

    def test_a_hole_shows_up_as_gap_and_as_coverage(self, tmp_path):
        days = ["2026-01-01", "2026-01-02", "2026-01-20"]
        result = audit_archive(self._archive(tmp_path, days), sources=[TEST_SOURCE])
        year = result.sources[0].years[0]
        assert year.days == 3 and year.expected_days == 20
        assert year.coverage_pct == 15.0
        assert year.max_gap_days == 18

    def test_event_sources_have_no_coverage_percentage(self, tmp_path):
        store = self._archive(
            tmp_path, ["2026-01-01", "2026-06-01"], entity=MARKET_ENTITY, source=EVENT_SOURCE
        )
        year = audit_archive(store, sources=[EVENT_SOURCE]).sources[0].years[0]
        assert year.coverage_pct is None  # un dia sin evento no es un dia sin dato
        assert year.days == 2

    def test_measured_history_is_what_the_archive_says(self, tmp_path):
        store = self._archive(tmp_path, ["2025-11-20", "2026-01-05"])
        source = audit_archive(store, sources=[TEST_SOURCE]).sources[0]
        assert source.measured_history_from == "2025-11-20"
        assert source.history_from is None  # el catalogo aun no lo declara
        assert source.declared_matches_measured is None
        assert [y.year for y in source.years] == [2025, 2026]

    def test_report_joins_both_audits(self, tmp_path):
        report = audit_module.audit_report(["BTC/USDT"], SignalStore(tmp_path))
        assert report["entities"]["coverage_pct"] == 100.0
        assert report["archive"]["n_sources"] == len(CATALOG)
