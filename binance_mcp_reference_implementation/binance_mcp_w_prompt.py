from __future__ import annotations

import csv
import io
import json
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from mcp.server.fastmcp import FastMCP
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================================================================
# Configuration
# =============================================================================

SERVER_NAME = "Binance MCP"
REQUEST_TIMEOUT_SECONDS = 10
DEFAULT_QUOTE_ASSET = "USDT"
MAX_LOG_RESOURCE_CHARS = 50_000

THIS_FOLDER = Path(__file__).resolve().parent
ACTIVITY_LOG_FILE = THIS_FOLDER / "activity.log"

BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24H_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"

# Fonte única de verdade para aliases.
SYMBOL_MAP: Dict[str, str] = {
    "btc": "BTCUSDT",
    "bitcoin": "BTCUSDT",
    "eth": "ETHUSDT",
    "ethereum": "ETHUSDT",
    "my_favorite": "ETCUSDT",
    "bnb": "BNBUSDT",
    "sol": "SOLUSDT",
    "solana": "SOLUSDT",
    "xrp": "XRPUSDT",
    "ada": "ADAUSDT",
    "doge": "DOGEUSDT",
    "dogecoin": "DOGEUSDT",
}


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger("binance_mcp")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    file_handler = logging.FileHandler(ACTIVITY_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


# =============================================================================
# MCP Server
# =============================================================================

mcp = FastMCP(SERVER_NAME)


# =============================================================================
# Exceptions
# =============================================================================

class BinanceMCPError(Exception):
    """Base exception for this MCP server."""


class SymbolValidationError(BinanceMCPError):
    """Raised when the input symbol is invalid."""


class BinanceAPIError(BinanceMCPError):
    """Raised when Binance request/response handling fails."""


class ActivityLogError(BinanceMCPError):
    """Raised when reading/writing the activity log fails."""


# =============================================================================
# HTTP Session
# =============================================================================

def build_http_session() -> Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "binance-mcp/2.0",
            "Accept": "application/json",
        }
    )

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


session = build_http_session()


# =============================================================================
# Helpers
# =============================================================================

def ensure_activity_log_file() -> None:
    try:
        ACTIVITY_LOG_FILE.touch(exist_ok=True)
    except OSError as exc:
        raise ActivityLogError(
            f"Unable to create or access activity log file: {ACTIVITY_LOG_FILE}"
        ) from exc


def utc_now_iso() -> str:
    from datetime import datetime, UTC

    return datetime.now(UTC).isoformat()


def safe_decimal_str(value: Any, field_name: str) -> str:
    try:
        return str(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise BinanceAPIError(
            f"Invalid numeric value for field '{field_name}': {value!r}"
        ) from exc


def log_activity(level: int, message: str, **extra: Any) -> None:
    if extra:
        message = f"{message} | data={json.dumps(extra, ensure_ascii=False, default=str)}"
    logger.log(level, message)


def make_success(action: str, symbol: Optional[str], data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "action": action,
        "symbol": symbol,
        "timestamp_utc": utc_now_iso(),
        "data": data,
    }


def make_error(action: str, symbol: Optional[str], error: Exception) -> Dict[str, Any]:
    return {
        "ok": False,
        "action": action,
        "symbol": symbol,
        "timestamp_utc": utc_now_iso(),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def build_symbol_map_csv() -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["crypto_name", "symbol"])
    for crypto_name, symbol in SYMBOL_MAP.items():
        writer.writerow([crypto_name, symbol])
    return output.getvalue()


def normalize_symbol(name: str) -> str:
    if not isinstance(name, str):
        raise SymbolValidationError("The symbol must be a string.")

    cleaned = name.strip().lower()
    if not cleaned:
        raise SymbolValidationError("The symbol cannot be empty.")

    if cleaned in SYMBOL_MAP:
        return SYMBOL_MAP[cleaned]

    upper = cleaned.upper().replace("-", "").replace("_", "").replace("/", "")
    if not upper:
        raise SymbolValidationError("The symbol cannot be empty after normalization.")

    if not upper.isalnum():
        raise SymbolValidationError(
            "The symbol contains invalid characters. Use only letters and numbers."
        )

    if upper.endswith(DEFAULT_QUOTE_ASSET):
        return upper

    return f"{upper}{DEFAULT_QUOTE_ASSET}"


def request_binance_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response: Response = session.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise BinanceAPIError(
            f"Request to Binance timed out after {REQUEST_TIMEOUT_SECONDS} seconds."
        ) from exc
    except requests.ConnectionError as exc:
        raise BinanceAPIError("Connection error while contacting Binance.") from exc
    except requests.RequestException as exc:
        raise BinanceAPIError(f"Unexpected request error: {exc}") from exc

    if response.status_code >= 400:
        body_preview = response.text[:500]
        raise BinanceAPIError(
            f"Binance returned HTTP {response.status_code}: {body_preview}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise BinanceAPIError("Binance returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise BinanceAPIError("Unexpected JSON structure returned by Binance.")

    return payload


def fetch_current_price(symbol: str) -> Dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)
    payload = request_binance_json(BINANCE_PRICE_URL, {"symbol": normalized_symbol})

    if "symbol" not in payload or "price" not in payload:
        raise BinanceAPIError(
            "Binance current price response is missing required fields."
        )

    result = {
        "symbol": payload["symbol"],
        "price": safe_decimal_str(payload["price"], "price"),
        "quote_asset": DEFAULT_QUOTE_ASSET,
        "source": BINANCE_PRICE_URL,
    }

    log_activity(
        logging.INFO,
        "Fetched current price successfully",
        symbol=normalized_symbol,
        price=result["price"],
    )

    return make_success("get_current_price", normalized_symbol, result)


def fetch_24h_ticker(symbol: str) -> Dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)
    payload = request_binance_json(BINANCE_24H_URL, {"symbol": normalized_symbol})

    required_fields = [
        "symbol",
        "lastPrice",
        "priceChange",
        "priceChangePercent",
        "highPrice",
        "lowPrice",
        "volume",
        "quoteVolume",
        "openTime",
        "closeTime",
    ]
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise BinanceAPIError(
            f"Binance 24h ticker response is missing required fields: {missing_fields}"
        )

    result = {
        "symbol": payload["symbol"],
        "last_price": safe_decimal_str(payload["lastPrice"], "lastPrice"),
        "price_change": safe_decimal_str(payload["priceChange"], "priceChange"),
        "price_change_percent": safe_decimal_str(
            payload["priceChangePercent"], "priceChangePercent"
        ),
        "high_price": safe_decimal_str(payload["highPrice"], "highPrice"),
        "low_price": safe_decimal_str(payload["lowPrice"], "lowPrice"),
        "volume": safe_decimal_str(payload["volume"], "volume"),
        "quote_volume": safe_decimal_str(payload["quoteVolume"], "quoteVolume"),
        "open_time_ms": payload["openTime"],
        "close_time_ms": payload["closeTime"],
        "source": BINANCE_24H_URL,
    }

    log_activity(
        logging.INFO,
        "Fetched 24h ticker successfully",
        symbol=normalized_symbol,
        price_change_percent=result["price_change_percent"],
    )

    return make_success("get_24h_price_change", normalized_symbol, result)


def fetch_exchange_info(symbol: str) -> Dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)
    payload = request_binance_json(BINANCE_EXCHANGE_INFO_URL, {"symbol": normalized_symbol})

    symbols = payload.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise BinanceAPIError("Exchange info response does not contain symbol details.")

    symbol_info = symbols[0]
    if not isinstance(symbol_info, dict):
        raise BinanceAPIError("Unexpected exchange info format for symbol.")

    result = {
        "symbol": symbol_info.get("symbol"),
        "status": symbol_info.get("status"),
        "base_asset": symbol_info.get("baseAsset"),
        "quote_asset": symbol_info.get("quoteAsset"),
        "base_asset_precision": symbol_info.get("baseAssetPrecision"),
        "quote_precision": symbol_info.get("quotePrecision"),
        "is_spot_trading_allowed": symbol_info.get("isSpotTradingAllowed"),
        "is_margin_trading_allowed": symbol_info.get("isMarginTradingAllowed"),
        "source": BINANCE_EXCHANGE_INFO_URL,
    }

    log_activity(
        logging.INFO,
        "Fetched exchange info successfully",
        symbol=normalized_symbol,
        status=result.get("status"),
    )

    return make_success("get_exchange_info", normalized_symbol, result)


def fetch_server_info() -> Dict[str, Any]:
    result = {
        "server_name": SERVER_NAME,
        "activity_log_file": str(ACTIVITY_LOG_FILE),
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "default_quote_asset": DEFAULT_QUOTE_ASSET,
        "known_aliases": sorted(SYMBOL_MAP.keys()),
        "transport": "stdio",
    }
    return make_success("server_info", None, result)


# =============================================================================
# Prompts
# =============================================================================

@mcp.prompt()
def executive_summary() -> str:
    """Prompt para resumo executivo de BTC e ETH."""
    return """
Get the current price and the 24-hour change data for BTCUSDT and ETHUSDT.

Prepare a concise executive summary that includes:
1. A two-sentence summary for Bitcoin.
2. A two-sentence summary for Ethereum.
3. Current price.
4. Absolute price change in the last 24 hours.
5. Percentage price change in the last 24 hours.

Use the tools:
- get_current_price
- get_24h_price_change

Symbols:
- btc / bitcoin -> BTCUSDT
- eth / ethereum -> ETHUSDT
"""


@mcp.prompt()
def crypto_summary(crypto: str) -> str:
    """Prompt para resumo de uma criptomoeda específica."""
    return f"""
Get the current price and the 24-hour change data for this crypto asset: {crypto}

Prepare a concise summary including:
1. Current price.
2. Absolute price change in the last 24 hours.
3. Percentage price change in the last 24 hours.
4. A short interpretation of whether the movement appears positive, negative, or neutral.

Use the tools:
- get_current_price
- get_24h_price_change

If needed, consult the resource:
- file://symbol_map.csv
"""


# =============================================================================
# Tools
# =============================================================================

@mcp.tool()
def get_current_price(symbol: str) -> Dict[str, Any]:
    """Get the current price of a crypto asset from Binance."""
    try:
        return fetch_current_price(symbol)
    except Exception as exc:
        log_activity(
            logging.ERROR,
            "Failed to fetch current price",
            symbol=symbol,
            error=str(exc),
        )
        return make_error("get_current_price", symbol, exc)


@mcp.tool()
def get_24h_price_change(symbol: str) -> Dict[str, Any]:
    """Get 24-hour ticker statistics of a crypto asset from Binance."""
    try:
        return fetch_24h_ticker(symbol)
    except Exception as exc:
        log_activity(
            logging.ERROR,
            "Failed to fetch 24h ticker",
            symbol=symbol,
            error=str(exc),
        )
        return make_error("get_24h_price_change", symbol, exc)


@mcp.tool()
def get_exchange_info(symbol: str) -> Dict[str, Any]:
    """Get exchange metadata for a symbol from Binance."""
    try:
        return fetch_exchange_info(symbol)
    except Exception as exc:
        log_activity(
            logging.ERROR,
            "Failed to fetch exchange info",
            symbol=symbol,
            error=str(exc),
        )
        return make_error("get_exchange_info", symbol, exc)


@mcp.tool()
def resolve_symbol(name: str) -> Dict[str, Any]:
    """Resolve a friendly name or alias to a Binance symbol."""
    try:
        normalized_symbol = normalize_symbol(name)
        result = {
            "input": name,
            "resolved_symbol": normalized_symbol,
            "source": "server_alias_map",
        }
        log_activity(
            logging.INFO,
            "Resolved symbol successfully",
            input=name,
            resolved_symbol=normalized_symbol,
        )
        return make_success("resolve_symbol", normalized_symbol, result)
    except Exception as exc:
        log_activity(
            logging.ERROR,
            "Failed to resolve symbol",
            input=name,
            error=str(exc),
        )
        return make_error("resolve_symbol", name, exc)


@mcp.tool()
def server_info() -> Dict[str, Any]:
    """Return local diagnostic information about this MCP server."""
    try:
        return fetch_server_info()
    except Exception as exc:
        log_activity(
            logging.ERROR,
            "Failed to fetch server info",
            error=str(exc),
        )
        return make_error("server_info", None, exc)


# =============================================================================
# Resources
# =============================================================================

@mcp.resource("file://activity.log")
def activity_log() -> str:
    """Expose the activity log as a readable MCP resource."""
    try:
        ensure_activity_log_file()
        content = ACTIVITY_LOG_FILE.read_text(encoding="utf-8")
        if len(content) > MAX_LOG_RESOURCE_CHARS:
            content = content[-MAX_LOG_RESOURCE_CHARS:]
        return content
    except OSError as exc:
        log_activity(logging.ERROR, "Failed to read activity log", error=str(exc))
        raise ActivityLogError("Unable to read activity log file.") from exc


@mcp.resource("file://symbol_map.csv")
def symbol_map_resource() -> str:
    """Expose the friendly-name-to-symbol map as CSV."""
    return build_symbol_map_csv()


@mcp.resource("resource://crypto/price/{symbol}")
def crypto_price_resource(symbol: str) -> str:
    """Expose current price as a resource."""
    result = get_current_price(symbol)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.resource("resource://crypto/24h/{symbol}")
def crypto_24h_resource(symbol: str) -> str:
    """Expose 24h ticker information as a resource."""
    result = get_24h_price_change(symbol)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.resource("resource://crypto/exchange-info/{symbol}")
def crypto_exchange_info_resource(symbol: str) -> str:
    """Expose exchange metadata as a resource."""
    result = get_exchange_info(symbol)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.resource("resource://server/info")
def server_info_resource() -> str:
    """Expose server metadata as a resource."""
    result = server_info()
    return json.dumps(result, ensure_ascii=False, indent=2)


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    try:
        ensure_activity_log_file()
        log_activity(logging.INFO, "Starting MCP server", server_name=SERVER_NAME)
        mcp.run(transport="stdio")
    except Exception as exc:
        log_activity(
            logging.CRITICAL,
            "Fatal error while starting MCP server",
            error=str(exc),
        )
        raise