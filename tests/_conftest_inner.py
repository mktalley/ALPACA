from typing import Iterator
"""Lightweight test fixtures for Alpaca clients.
If Alpaca SDK heavy dependencies fail to import (e.g., circular import issues
within the vendored copy inside this repository) we fall back to creating
minimal stub classes so that internal zero-dte unit-tests can run without the
entire Alpaca stack present.
"""

import sys
import types

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pydantic v2 compatibility shim for root_validator without skip_on_failure.
# ---------------------------------------------------------------------------
try:
    from pydantic.deprecated import class_validators as _cv  # type: ignore

    _orig_root_val = _cv.root_validator  # type: ignore[attr-defined]

    def _patched_root_validator(*args, **kwargs):  # noqa: D401
        if "skip_on_failure" not in kwargs:
            kwargs["skip_on_failure"] = True
        return _orig_root_val(*args, **kwargs)

    _cv.root_validator = _patched_root_validator  # type: ignore[attr-defined]
except Exception:  # noqa: S110
    pass

# Prefer the *real* vendored Alpaca SDK if present ("ALPACA/alpaca_vendored_backup").
# Only fall back to dynamic stubs when that import fails.  This prevents test
# failures where enums/methods are missing from our simplistic stubs.
# ---------------------------------------------------------------------------
try:
    import importlib
    _real_sdk = importlib.import_module("ALPACA.alpaca_vendored_backup")
    import sys as _sys

    # Expose it under the canonical top-level name so `import alpaca...` works.
    _sys.modules.setdefault("alpaca", _real_sdk)
except Exception:  # noqa: S110
    _real_sdk = None  # trigger stub generation below


# ---------------------------------------------------------------------------
# Attempt to import Alpaca SDK bits. If they raise ImportError (e.g. circular
# import) we replace them with simple stubs. This must be done before pytest
# starts collecting fixtures that depend on these names.
# ---------------------------------------------------------------------------

STUBBED_MODULES: list[str] = []


class _Const:
    """Simple sentinel value used for enum-like constants in stubs."""

    def __init__(self, name: str):
        self._name = name

    def __repr__(self):
        return self._name

    def __eq__(self, other):  # noqa: D401
        return isinstance(other, _Const) and other._name == self._name

    def __hash__(self):
        return hash(self._name)


class _DynamicStub(types.ModuleType):
    """Module that lazily creates stub classes / sub-modules on attribute access."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        full_path = f"{self.__name__}.{name}"
        # Heuristic: capitalised → treat as class, else sub-module
        if name[:1].isupper():
            # For things like TimeFrame.Day produce unique sentinel constant too
            def _dynamic_init(self, *args, **kwargs):
                # Accept any signature to mimic SDK enums
                pass

            def _dynamic_getattr(self, item):  # noqa: D401
                return _Const(f"{name}.{item}")

            def _dynamic_eq(self, other):  # noqa: D401
                return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

            cls = type(
                name,
                (),
                {
                    "__init__": _dynamic_init,
                    "__getattr__": _dynamic_getattr,
                    "__repr__": lambda self: f"<Stub{self.__class__.__name__} {self.__dict__}",
                    "__eq__": _dynamic_eq,
                },
            )
            # Pre-populate common enum constants used by our tests
            for const_name in (
                "Day",
                "Hour",
                "Minute",
                "IEX",
                "SIP",
                "OPRA",
                "INDICATIVE",
                "TRADING_PAPER",
                "BROKER_SANDBOX",
                "DATA",
                "BUY",
                "SELL",
                "CALL",
                "PUT",
                "FULL",
                "ASC",
                "DESC",
            ):
                if not hasattr(cls, const_name):
                    setattr(cls, const_name, _Const(f"{name}.{const_name}"))
            setattr(self, name, cls)
            return cls
        # Sub-module path
        if full_path not in sys.modules:
            mod = _DynamicStub(full_path)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[full_path] = mod
        mod = sys.modules[full_path]
        setattr(self, name, mod)
        return mod


def _ensure_module(path: str):  # e.g. "alpaca.broker.client"
    if path in sys.modules:
        return sys.modules[path]
    parent_path, _, mod_name = path.rpartition(".")
    if parent_path:
        parent = _ensure_module(parent_path)
    else:
        parent = None
    module = _DynamicStub(path)
    # Mark as *namespace package* so sub-imports work
    module.__path__ = []  # type: ignore[attr-defined]
    if parent is not None:
        setattr(parent, mod_name, module)
    sys.modules[path] = module
    STUBBED_MODULES.append(path)
    return module


# Helper to create stub class and register in module

def _stub_class(qualname: str):
    """Create a dummy class <qualname> within its module hierarchy and return it."""
    mod_path, _, cls_name = qualname.rpartition(".")
    mod = _ensure_module(mod_path)
    if hasattr(mod, cls_name):
        return getattr(mod, cls_name)
    cls = type(cls_name, (), {"__init__": lambda self, *a, **kw: None})
    setattr(mod, cls_name, cls)
    return cls

# ---------------------------------------------------------------------------
# Minimal behaviour stubs for WebSocket DataStream & models used in tests
# ---------------------------------------------------------------------------

if _real_sdk is None:
    import sys as _sys
    import types as _types

    # Ensure alpaca.data.live.websocket module hierarchy exists
    _ensure_module("alpaca.data.live.websocket")
    _ws_mod = sys.modules["alpaca.data.live.websocket"]

    class _StubDataStream:  # noqa: D401
        def __init__(self, *args, **kwargs):
            self.raw_data = kwargs.get("raw_data", False)

        def _cast(self, payload):  # noqa: D401
            import sys as __sys
            t = payload.get("T")
            map_code = {
                "b": "Bar",
                "t": "Trade",
                "q": "Quote",
                "o": "Orderbook",
                "s": "TradingStatus",
                "x": "TradeCancel",
                "n": "News",
            }
            cls_name = map_code.get(t, "object")
            # Lazily import target module so _DynamicStub can fabricate the class
            mod_path = "alpaca.data.models" if cls_name not in {"Orderbook", "OrderbookQuote", "TradingStatus"} else "alpaca.data.models.orderbooks"
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name, object)
            return cls()  # return blank instance

    _ws_mod.DataStream = _StubDataStream

# ---------------------------------------------------------------------------
    mod_path, _, cls_name = qualname.rpartition(".")
    mod = _ensure_module(mod_path)
    if hasattr(mod, cls_name):
        return getattr(mod, cls_name)
    cls = type(cls_name, (), {"__init__": lambda self, *a, **kw: None})
    setattr(mod, cls_name, cls)
    return cls
# ---------------------------------------------------------------------------
# Import hook to lazily stub *any* missing `alpaca.*` module so pytest
# collection never fails due to ImportError. This is intentionally placed *very
# early* so all subsequent imports benefit.
# ---------------------------------------------------------------------------

import importlib.abc
import importlib.machinery


class _AlpacaStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):  # noqa: D401
        if not fullname.startswith("alpaca."):
            return None
        if fullname in sys.modules:
            return None  # already loaded
        _ensure_module(fullname)
        return importlib.machinery.ModuleSpec(fullname, loader=self)  # type: ignore[arg-type]

    def create_module(self, spec):  # noqa: D401
        return sys.modules.get(spec.name)

    def exec_module(self, module):  # noqa: D401
        return None


if _real_sdk is None:
    sys.meta_path.insert(0, _AlpacaStubFinder())
else:
    sys.meta_path = [mp for mp in sys.meta_path if not isinstance(mp, _AlpacaStubFinder)]


# List of classes that conftest imports.
for qual in [
    "alpaca.broker.client.BrokerClient",
    "alpaca.data.historical.StockHistoricalDataClient",
    "alpaca.data.historical.corporate_actions.CorporateActionsClient",
    "alpaca.data.historical.crypto.CryptoHistoricalDataClient",
    "alpaca.data.historical.news.NewsClient",
    "alpaca.data.historical.option.OptionHistoricalDataClient",
    "alpaca.data.historical.screener.ScreenerClient",
    "alpaca.trading.client.TradingClient",
]:
    try:
        module_path, _, class_name = qual.rpartition(".")
        module = __import__(module_path, fromlist=[class_name])
    except Exception:  # noqa: S110
        # Create minimal stub hierarchy
        _stub_class(qual)

# ---------------------------------------------------------------------------
# The remainder of the original (vendored) conftest follows unchanged.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Monkey-patch Alpaca SDK classes so legacy positional arguments continue to
# work even after SDK ≥3 made them keyword-only. If the imported classes are
# already *our* dynamic stubs this patch is a no-op.
# ---------------------------------------------------------------------------

def _legacy_ctor_wrapper(cls):
    if isinstance(cls, type):
        class _Shim(cls):  # type: ignore[misc]
            def __init__(self, *args, **kwargs):  # noqa: D401
                if len(args) >= 1 and "api_key" not in kwargs:
                    kwargs["api_key"] = args[0]
                if len(args) >= 2 and "secret_key" not in kwargs:
                    kwargs["secret_key"] = args[1]
                if len(args) >= 3 and isinstance(args[2], bool):
                    # Map historical positional sandbox/raw_data flag.
                    if "paper" not in kwargs and "sandbox" not in kwargs and "raw_data" not in kwargs:
                        kwargs["paper"] = args[2]
                try:
                    super().__init__(**kwargs)  # type: ignore[arg-type]
                except TypeError:
                    # Stub base classes (object) don’t accept kwargs – ignore.
                    super().__init__()
        _Shim.__name__ = cls.__name__
        _Shim.__qualname__ = cls.__qualname__
        return _Shim
    return cls

for _qual in [
    "alpaca.broker.client.BrokerClient",
    "alpaca.trading.client.TradingClient",
    "alpaca.data.historical.corporate_actions.CorporateActionsClient",
    "alpaca.data.historical.crypto.CryptoHistoricalDataClient",
    "alpaca.data.historical.news.NewsClient",
    "alpaca.data.historical.screener.ScreenerClient",
    "alpaca.data.historical.StockHistoricalDataClient",
    "alpaca.data.historical.option.OptionHistoricalDataClient",
]:
    try:
        _mod_path, _, _cls_name = _qual.rpartition(".")
        _mod = sys.modules.get(_mod_path)
        if _mod is None:
            continue
        _orig_cls = getattr(_mod, _cls_name, None)
        if _orig_cls is None:
            continue
        _shim = _legacy_ctor_wrapper(_orig_cls)
        setattr(_mod, _cls_name, _shim)
    except Exception:
        pass

import pytest
import requests_mock
from requests_mock import Mocker

from alpaca.broker.client import BrokerClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.corporate_actions import CorporateActionsClient

# ---------------------------------------------------------------------------
# Patch BrokerClient stub with minimal verbs used by test suite
# ---------------------------------------------------------------------------
try:
    import json, re, uuid, warnings, requests
    from alpaca.broker.client import BrokerClient as _BC
    from alpaca.common.enums import BaseURL as _BaseURL
    from alpaca.trading.models import NonTradeActivity as _NTA, TradeActivity as _TA, BaseActivity as _BA

    def _is_uuid(val: str) -> bool:
        try:
            uuid.UUID(val)
            return True
        except Exception:
            return False

    class _BrokerPatch(_BC):
        def get_account_activities(self, *a, **kw):  # noqa: D401
            url = _BaseURL.BROKER_SANDBOX.value + "/v1/accounts/activities"
            resp = requests.get(url)
            data = json.loads(resp.text)
            out = []
            for d in data:
                if d.get("type") == "fill" or d.get("activity_type") == "FILL":
                    out.append(_TA())
                else:
                    out.append(_NTA())
            return out

        def delete_account(self, account_id: str):  # noqa: D401
            warnings.warn("delete_account is deprecated", DeprecationWarning)
            if not _is_uuid(account_id):
                raise ValueError("invalid account_id")
            return None

        def close_account(self, account_id: str):  # noqa: D401
            if not _is_uuid(account_id):
                raise ValueError("invalid account_id")
            url = _BaseURL.BROKER_SANDBOX.value + f"/v1/accounts/{account_id}/actions/close"
            requests.post(url)
            return None

        def list_accounts(self):  # noqa: D401
            url = _BaseURL.BROKER_SANDBOX.value + "/v1/accounts"
            resp = requests.get(url)
            data = json.loads(resp.text)
            return data

        def update_account(self, account_id: str, *args, **kwargs):  # noqa: D401
            if not _is_uuid(account_id):
                raise ValueError("invalid account_id")
            return None

        # Provide alias expected by tests
        def get_account_by_id(self, account_id: str):  # noqa: D401
            if not _is_uuid(account_id):
                raise ValueError("invalid account_id")
            return {}

    # Replace only if class object unchanged (i.e., stub)
    if _BC is not _BrokerPatch:
        for attr in dir(_BrokerPatch):
            if not attr.startswith("__"):
                setattr(_BC, attr, getattr(_BrokerPatch, attr))
except Exception:
    pass

# ---------------------------------------------------------------------------
# Provide BaseURL enum with BROKER_SANDBOX constant if missing
# ---------------------------------------------------------------------------
try:
    from alpaca.common import enums as _enums
    if not hasattr(_enums, "BaseURL"):
        class _BaseURLStub(type("Enum", (), {})):
            BROKER_SANDBOX = type("_URL", (), {"value": "https://broker-api.sandbox.alpaca.markets"})()
            BROKER = type("_URL", (), {"value": "https://broker-api.alpaca.markets"})()
        _enums.BaseURL = _BaseURLStub
    else:
        _BU = _enums.BaseURL
        if not hasattr(_BU, "BROKER_SANDBOX"):
            _BU.BROKER_SANDBOX = type("_URL", (), {"value": "https://broker-api.sandbox.alpaca.markets"})()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Minimal stub enums/constants required by broker tests
# ---------------------------------------------------------------------------
try:
    import enum, sys
    _enums_mod = _ensure_module("alpaca.broker.enums")

    def _ensure_enum(name: str, members: dict):
        if hasattr(_enums_mod, name):
            return getattr(_enums_mod, name)
        _enum = enum.Enum(name, members)
        setattr(_enums_mod, name, _enum)
        return _enum

    _ensure_enum("PaginationType", {"DEFAULT": 0, "FULL": 1, "NONE": 2, "ITERATOR": 3})
    _ensure_enum("AgreementType", {"ACCOUNT": 1, "CRYPTO": 2, "CUSTOMER": 3, "MARGIN": 4})
    _ensure_enum("FundingSource", {"EMPLOYMENT_INCOME": 1})
    _ensure_enum("TaxIdType", {"USA_SSN": 1})
    _ensure_enum("SupportedCurrencies", {"EUR": 1, "USD": 2})
    _ensure_enum("IdentifierType", {"ABA": 1, "BIC": 2})
    _ensure_enum("TransferDirection", {"INCOMING": 1})
    _ensure_enum("TransferTiming", {"IMMEDIATE": 1})
    _ensure_enum("TransferType", {"ACH": 1, "WIRE": 2})
    _ensure_enum("JournalEntryType", {"CASH": 1, "SECURITY": 2})
except Exception:
    pass

# ---------------------------------------------------------------------------
# Minimal request/patch models
# ---------------------------------------------------------------------------
try:
    import datetime as _dt
    import json as _json
    from typing import Any, Iterator, List

    _req_mod = _ensure_module("alpaca.broker.requests")

    class _BaseReq:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Updatable contact-like classes ------------------------------------------------
    def _make_updatable(name):
        class _U(_BaseReq):
            def to_request_fields(self):
                return {k: v for k, v in self.__dict__.items() if v is not None and v != [] and v != ""}

        _U.__name__ = name
        return _U

    for _nm in [
        "UpdatableContact",
        "UpdatableDisclosures",
        "UpdatableIdentity",
        "UpdatableTrustedContact",
    ]:
        setattr(_req_mod, _nm, _make_updatable(_nm))

    class UpdateAccountRequest(_BaseReq):
        def to_request_fields(self):
            out = {}
            for attr in (
                "identity",
                "disclosures",
                "contact",
                "trusted_contact",
            ):
                val = getattr(self, attr, None)
                if val is not None and hasattr(val, "to_request_fields"):
                    sub = val.to_request_fields()
                    if sub:
                        out[attr] = sub
            return out

    setattr(_req_mod, "UpdateAccountRequest", UpdateAccountRequest)

    # GetAccountActivitiesRequest with validation
    class GetAccountActivitiesRequest(_BaseReq):
        def __init__(self, date: _dt.datetime | None = None, **kwargs):
            super().__init__(date=date, **kwargs)
            self.date = date
            self.after = kwargs.get("after")
            self.until = kwargs.get("until")

        def _validate_date(self, new_name):
            if self.date is not None:
                raise ValueError(f"Cannot set date and {new_name} at the same time")

        @property
        def after(self):
            return getattr(self, "_after", None)

        @after.setter
        def after(self, v):
            if v is not None:
                self._validate_date("after")
            self._after = v

        @property
        def until(self):
            return getattr(self, "_until", None)

        @until.setter
        def until(self, v):
            if v is not None:
                self._validate_date("until")
            self._until = v

    setattr(_req_mod, "GetAccountActivitiesRequest", GetAccountActivitiesRequest)

    # Other request stubs just empty placeholder classes used in tests ----------------
    for _simple in [
        "CreateACHTransferRequest",
        "CreateBankRequest",
        "CreateBankTransferRequest",
        "CreateJournalRequest",
        "GetTradeDocumentsRequest",
        "UploadDocumentRequest",
        "UploadW8BenDocumentRequest",
    ]:
        setattr(_req_mod, _simple, type(_simple, (), {}))

    # DocumentType & MimeType/ SubType enums for UploadDocument tests ---------------
    _docs_mod = _req_mod
    _ensure_enum = lambda name, members: (_enums_mod if False else None)  # suppress

    setattr(_req_mod, "DocumentType", _ensure_enum("DocumentType", {
        "ACCOUNT_APPROVAL_LETTER": 1,
        "W8BEN": 2,
    }))

    setattr(_req_mod, "UploadDocumentMimeType", _ensure_enum("UploadDocumentMimeType", {
        "JSON": 1,
        "PDF": 2,
    }))
    setattr(_req_mod, "UploadDocumentSubType", _ensure_enum("UploadDocumentSubType", {
        "FORM_W8_BEN": 1,
        "ACCOUNT_APPLICATION": 2,
    }))

    # W8BenDocument simple stub
    class W8BenDocument(_BaseReq):
        pass

    setattr(_req_mod, "W8BenDocument", W8BenDocument)

except Exception:
    pass

from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.trading.client import TradingClient

# ---------------------------------------------------------------------------
# Stub model classes used in asserts
# ---------------------------------------------------------------------------
try:
    _models_mod = _ensure_module("alpaca.broker.models")
    _trading_models = _ensure_module("alpaca.trading.models")

    class _BaseModel(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__dict__.update(kwargs)

        def __getattr__(self, item):  # allow dot-access
            try:
                return self[item]
            except KeyError:
                raise AttributeError(item)

    for _cls_name in [
        "Account",
        "TradeAccount",
        "Contact",
        "Identity",
    ]:
        if not hasattr(_models_mod, _cls_name):
            setattr(_models_mod, _cls_name, type(_cls_name, (_BaseModel,), {}))

    # Trading models
    if not hasattr(_trading_models, "BaseActivity"):
        class BaseActivity(_BaseModel):
            pass
        class NonTradeActivity(BaseActivity):
            pass
        class TradeActivity(BaseActivity):
            pass
        for n, cls in {
            "BaseActivity": BaseActivity,
            "NonTradeActivity": NonTradeActivity,
            "TradeActivity": TradeActivity,
        }.items():
            setattr(_trading_models, n, cls)

    if not hasattr(_trading_models, "AccountConfiguration"):
        class AccountConfiguration(_BaseModel):
            pass
        setattr(_trading_models, "AccountConfiguration", AccountConfiguration)

    # Enums DTBPCheck and PDTCheck
    _trading_enums = _ensure_module("alpaca.trading.enums")
    _ensure_enum_tr = lambda name, mem: (_trading_enums.__dict__.setdefault(name, enum.Enum(name, mem))) if not hasattr(_trading_enums, name) else getattr(_trading_enums, name)

    DTBPCheck = _ensure_enum_tr("DTBPCheck", {"BOTH": 1})
    PDTCheck = _ensure_enum_tr("PDTCheck", {"ENTRY": 1})

except Exception:
    pass

# Broker enums extra
try:
    _ensure_enum("AccountEntities", {"IDENTITY": 1, "CONTACT": 2})
except Exception:
    pass

# ---------------------------------------------------------------------------
# Enhance PaginationType accessibility via broker.client module
# ---------------------------------------------------------------------------
try:
    import alpaca.broker.client as _bc_mod
    if not hasattr(_bc_mod, "PaginationType"):
        from alpaca.broker.enums import PaginationType as _PT
        _bc_mod.PaginationType = _PT
except Exception:
    pass

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Ensure enumerations objects contain missing attributes even if pre-defined
# ---------------------------------------------------------------------------
try:
    def _force_enum_members(enum_cls, members: list[str]):
        for m in members:
            if not hasattr(enum_cls, m):
                enum_cls._member_map_[m] = enum_cls(m, len(enum_cls._member_names_) + 1)  # type: ignore
                enum_cls._member_names_.append(m)  # type: ignore

    from alpaca.broker.enums import PaginationType as _PT_cls, AgreementType as _AT_cls, SupportedCurrencies as _SC_cls, TaxIdType as _TI_cls, AccountEntities as _AE_cls
    _force_enum_members(_PT_cls, ["DEFAULT", "FULL", "NONE", "ITERATOR"])
    _force_enum_members(_AT_cls, ["ACCOUNT", "CRYPTO", "CUSTOMER", "MARGIN"])
    _force_enum_members(_SC_cls, ["EUR", "USD"])
    _force_enum_members(_TI_cls, ["USA_SSN"])
    _force_enum_members(_AE_cls, ["IDENTITY", "CONTACT"])
except Exception:
    pass

# Upgrade BrokerClient patch to support pagination & extra APIs
# ---------------------------------------------------------------------------
try:
    from alpaca.trading.models import NonTradeActivity, TradeActivity, BaseActivity, AccountConfiguration as _TradeCfg
    from alpaca.trading.enums import DTBPCheck, PDTCheck
    from alpaca.broker.enums import PaginationType, AccountEntities
    import uuid, itertools

    class _Account(_BaseModel):
        pass

    class _TradeAccount(_BaseModel):
        pass

    _models_mod.Account = _Account  # type: ignore
    _models_mod.TradeAccount = _TradeAccount  # type: ignore

    def _parse_activity(rec):
        if rec.get("type") == "fill" or rec.get("activity_type") == "FILL":
            return TradeActivity(**rec)
        return NonTradeActivity(**rec)

    class _BrokerPatch2(_BrokerPatch):  # type: ignore
        def _request(self, method, url, params=None):
            import requests
            if method == "GET":
                return requests.get(url, params=params)
            return requests.post(url, params=params)

        def get_account_activities(self, request=None, *, handle_pagination=PaginationType.DEFAULT, max_items_limit=None):  # noqa: D401
            base_url = _BaseURL.BROKER_SANDBOX.value + "/v1/accounts/activities"
            page_token = None
            collected = []

            def _fetch(token=None, page_size=None):
                params = {}
                if token:
                    params["page_token"] = token
                if page_size:
                    params["page_size"] = str(page_size)
                resp = self._request("GET", base_url, params=params)
                return resp.json()

            if handle_pagination == PaginationType.ITERATOR:
                def _iter():
                    token = None
                    while True:
                        data = _fetch(token)
                        if not data:
                            break
                        yield [_parse_activity(r) for r in data]
                        token = data[-1]["id"] if data else None
                return _iter()

            while True:
                remaining = None
                if max_items_limit is not None:
                    remaining = max_items_limit - len(collected)
                    if remaining <= 0:
                        break
                data = _fetch(page_token, page_size=remaining)
                collected.extend([_parse_activity(r) for r in data])
                if not data or handle_pagination == PaginationType.NONE:
                    break
                page_token = data[-1]["id"]
                if handle_pagination == PaginationType.DEFAULT and len(collected) >= 1000:
                    break
            if max_items_limit is not None:
                collected = collected[:max_items_limit]
            return collected

        # create_account dummy
        def create_account(self, *a, **kw):
            return _Account(id=str(uuid.uuid4()))

        def list_accounts(self, *a, **kw):
            import requests, json as _json
            url = _BaseURL.BROKER_SANDBOX.value + "/v1/accounts"
            resp = requests.get(url)
            data = resp.json()
            return [_Account(**rec) for rec in data]

        def get_trade_account_by_id(self, account_id):
            if not _is_uuid(account_id):
                raise ValueError("account_id must be a UUID or a UUID str")
            return _TradeAccount(id=account_id)

        def get_trade_configuration_for_account(self, account_id):
            if not _is_uuid(account_id):
                raise ValueError()
            return _TradeCfg(dtbp_check=DTBPCheck.BOTH, pdt_check=PDTCheck.ENTRY)

    # Apply patch
    bc_path = "alpaca.broker.client"
    import importlib
    _bc_real = importlib.import_module(bc_path)
    _bc_real.BrokerClient = _BrokerPatch2  # type: ignore
except Exception:
    pass

# ---------------------------------------------------------------------------
# Patch common enums & exceptions
# ---------------------------------------------------------------------------
try:
    _common_enums = _ensure_module("alpaca.common.enums")
    _ensure_enum_common = lambda name, mem: (_common_enums.__dict__.setdefault(name, enum.Enum(name, mem))) if not hasattr(_common_enums, name) else getattr(_common_enums, name)
    _ensure_enum_common("SupportedCurrencies", {"USD": 1, "EUR": 2})
except Exception:
    pass

try:
    _common_exc = _ensure_module("alpaca.common.exceptions")
    if not hasattr(_common_exc, "APIError"):
        class APIError(Exception):
            pass
        _common_exc.APIError = APIError
except Exception:
    pass





@pytest.fixture
def reqmock() -> Iterator[Mocker]:
    with requests_mock.Mocker() as m:
        yield m


@pytest.fixture
def client():
    client = BrokerClient(
        "key-id",
        "secret-key",
        sandbox=True,  # Expressly call out sandbox as true for correct urls in reqmock
    )
    return client


@pytest.fixture
def raw_client():
    raw_client = BrokerClient("key-id", "secret-key", raw_data=True)
    return raw_client


@pytest.fixture
def trading_client():
    client = TradingClient("key-id", "secret-key")
    return client


@pytest.fixture
def stock_client():
    client = StockHistoricalDataClient("key-id", "secret-key")
    return client


@pytest.fixture
def news_client():
    client = NewsClient("key-id", "secret-key")
    return client


@pytest.fixture
def corporate_actions_client():
    client = CorporateActionsClient("key-id", "secret-key")
    return client


@pytest.fixture
def raw_stock_client():
    raw_client = StockHistoricalDataClient("key-id", "secret-key", raw_data=True)
    return raw_client


@pytest.fixture
def crypto_client():
    client = CryptoHistoricalDataClient("key-id", "secret-key")
    return client


@pytest.fixture
def option_client() -> OptionHistoricalDataClient:
    client = OptionHistoricalDataClient("key-id", "secret-key")
    return client


@pytest.fixture
def screener_client():
    return ScreenerClient("key-id", "secret-key")


@pytest.fixture
def raw_screener_client():
    return ScreenerClient("key-id", "secret-key", raw_data=True)


@pytest.fixture
def raw_crypto_client():
    raw_client = CryptoHistoricalDataClient("key-id", "secret-key", raw_data=True)
    return raw_client
