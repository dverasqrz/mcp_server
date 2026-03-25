import datetime as dt
import json
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from mcp.server.fastmcp import FastMCP

# ============================================================================
# Configuration
# ============================================================================

SERVER_NAME = "Binance MCP"
REQUEST_TIMEOUT_SECONDS = 10
DEFAULT_QUOTE_ASSET = "USDT"

THIS_FOLDER = Path(__file__).resolve().parent
ACTIVITY_LOG_FILE = THIS_FOLDER / "activity.log"

BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24H_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"

SYMBOL_ALIASES = {
    "bitcoin": "BTCUSDT",
    "btc": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "eth": "ETHUSDT",
    "bnb": "BNBUSDT",
    "sol": "SOLUSDT",
    "solana": "SOLUSDT",
    "xrp": "XRPUSDT",
    "ada": "ADAUSDT",
    "doge": "DOGEUSDT",
    "dogecoin": "DOGEUSDT",
}

# ============================================================================
# Logging setup
# ============================================================================

logger = logging.getLogger("binance_mcp")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    file_handler = logging.FileHandler(ACTIVITY_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Force UTC timestamps in logs
logging.Formatter.converter = time_converter = lambda *args: dt.datetime.now(
    dt.UTC
).timetuple()

# ============================================================================
# MCP server
# ============================================================================

mcp = FastMCP(SERVER_NAME)

# Reusable HTTP session
session = requests.Session()
session.headers.update(
    {
        "User-Agent": "binance-mcp/1.0",
        "Accept": "application/json",
    }
)

# ============================================================================
# Custom exceptions
# ============================================================================


class BinanceMCPError(Exception):
    """Base exception for the Binance MCP server."""


class SymbolValidationError(BinanceMCPError):
    """Raised when the symbol provided by the user is invalid."""


class BinanceAPIError(BinanceMCPError):
    """Raised when Binance returns an invalid or unsuccessful response."""


class ActivityLogError(BinanceMCPError):
    """Raised when the activity log cannot be accessed."""


# ============================================================================
# Utility helpers
# ============================================================================


def utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return dt.datetime.now(dt.UTC).isoformat()


def ensure_activity_log_file() -> None:
    """Ensure the activity log file exists."""
    try:
        ACTIVITY_LOG_FILE.touch(exist_ok=True)
    except OSError as exc:
        raise ActivityLogError(
            f"Unable to create or access activity log file: {ACTIVITY_LOG_FILE}"
        ) from exc


def normalize_symbol(name: str) -> str:
    """
    Normalize a human-friendly asset name to a Binance trading pair symbol.

    Rules:
    - Known aliases like btc/bitcoin -> BTCUSDT
    - If already ends with quote asset (e.g. BTCUSDT), preserve it
    - Otherwise append USDT by default
    """
    if not isinstance(name, str):
        raise SymbolValidationError("Symbol must be a string.")

    cleaned = name.strip().lower()
    if not cleaned:
        raise SymbolValidationError("Symbol cannot be empty.")

    if cleaned in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[cleaned]

    upper = cleaned.upper().replace("-", "").replace("_", "").replace("/", "")

    if not upper.isalnum():
        raise SymbolValidationError(
            "Symbol contains invalid characters. Use letters/numbers only."
        )

    if upper.endswith(DEFAULT_QUOTE_ASSET):
        return upper

    return f"{upper}{DEFAULT_QUOTE_ASSET}"


def safe_decimal(value: Any, field_name: str) -> str:
    """
    Validate and normalize a numeric value to string using Decimal.

    Returns string to preserve precision in JSON responses.
    """
    try:
        return str(Decimal(str(value)))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BinanceAPIError(
            f"Invalid numeric value received from Binance for '{field_name}': {value}"
        ) from exc


def log_activity(level: int, message: str, **extra: Any) -> None:
    """Write structured information to logs."""
    if extra:
        message = f"{message} | data={json.dumps(extra, ensure_ascii=False, default=str)}"
    logger.log(level, message)


def request_binance_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute an HTTP GET request to Binance and return parsed JSON.

    Raises:
        BinanceAPIError: for network, HTTP, or payload issues.
    """
    try:
        response = session.get(
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
        raise BinanceAPIError(f"Unexpected HTTP error while contacting Binance: {exc}") from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text[:500]
        raise BinanceAPIError(
            f"Binance returned HTTP {response.status_code}: {response_text}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise BinanceAPIError("Binance returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise BinanceAPIError("Unexpected JSON structure returned by Binance.")

    return data


def success_response(action: str, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Standard success payload."""
    return {
        "ok": True,
        "action": action,
        "symbol": symbol,
        "timestamp_utc": utc_now_iso(),
        "data": data,
    }


def error_response(action: str, symbol: Optional[str], error: Exception) -> Dict[str, Any]:
    """Standard error payload."""
    return {
        "ok": False,
        "action": action,
        "symbol": symbol,
        "timestamp_utc": utc_now_iso(),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


# ============================================================================
# Core business functions
# ============================================================================


def fetch_current_price(symbol: str) -> Dict[str, Any]:
    """
    Fetch the current price for a symbol from Binance.
    """
    normalized_symbol = normalize_symbol(symbol)
    payload = request_binance_json(BINANCE_PRICE_URL, {"symbol": normalized_symbol})

    if "price" not in payload or "symbol" not in payload:
        raise BinanceAPIError(
            "Binance response for current price does not contain expected fields."
        )

    result = {
        "symbol": payload["symbol"],
        "price": safe_decimal(payload["price"], "price"),
        "quote_asset": DEFAULT_QUOTE_ASSET,
        "source": "Binance /api/v3/ticker/price",
    }

    log_activity(
        logging.INFO,
        "Fetched current price successfully",
        symbol=normalized_symbol,
        price=result["price"],
    )

    return success_response("get_current_price", normalized_symbol, result)


def fetch_24h_ticker(symbol: str) -> Dict[str, Any]:
    """
    Fetch 24h ticker statistics for a symbol from Binance.
    """
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

    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise BinanceAPIError(
            f"Binance response for 24h ticker is missing required fields: {missing}"
        )

    result = {
        "symbol": payload["symbol"],
        "last_price": safe_decimal(payload["lastPrice"], "lastPrice"),
        "price_change": safe_decimal(payload["priceChange"], "priceChange"),
        "price_change_percent": safe_decimal(
            payload["priceChangePercent"], "priceChangePercent"
        ),
        "high_price": safe_decimal(payload["highPrice"], "highPrice"),
        "low_price": safe_decimal(payload["lowPrice"], "lowPrice"),
        "volume": safe_decimal(payload["volume"], "volume"),
        "quote_volume": safe_decimal(payload["quoteVolume"], "quoteVolume"),
        "open_time_ms": payload["openTime"],
        "close_time_ms": payload["closeTime"],
        "source": "Binance /api/v3/ticker/24hr",
    }

    log_activity(
        logging.INFO,
        "Fetched 24h ticker successfully",
        symbol=normalized_symbol,
        last_price=result["last_price"],
        price_change_percent=result["price_change_percent"],
    )

    return success_response("get_24h_ticker", normalized_symbol, result)


def fetch_server_info() -> Dict[str, Any]:
    """
    Return local server metadata useful for diagnostics.
    """
    return {
        "ok": True,
        "action": "server_info",
        "timestamp_utc": utc_now_iso(),
        "data": {
            "server_name": SERVER_NAME,
            "activity_log_file": str(ACTIVITY_LOG_FILE),
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "default_quote_asset": DEFAULT_QUOTE_ASSET,
            "known_aliases": sorted(SYMBOL_ALIASES.keys()),
            "transport": "stdio",
        },
    }


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
def get_current_price(symbol: str) -> Dict[str, Any]:
    """
    Get the current price of a crypto asset from Binance.

    Args:
        symbol: Asset name or symbol, such as BTC, bitcoin, ETHUSDT, sol

    Returns:
        A structured JSON-compatible dict with current price information.
    """
    try:
        return fetch_current_price(symbol)
    except Exception as exc:
        log_activity(logging.ERROR, "Failed to fetch current price", symbol=symbol, error=str(exc))
        return error_response("get_current_price", symbol, exc)


@mcp.tool()
def get_24h_price_change(symbol: str) -> Dict[str, Any]:
    """
    Get the 24-hour ticker statistics of a crypto asset from Binance.

    Args:
        symbol: Asset name or symbol, such as BTC, bitcoin, ETHUSDT, sol

    Returns:
        A structured JSON-compatible dict with 24h price statistics.
    """
    try:
        return fetch_24h_ticker(symbol)
    except Exception as exc:
        log_activity(logging.ERROR, "Failed to fetch 24h ticker", symbol=symbol, error=str(exc))
        return error_response("get_24h_price_change", symbol, exc)


@mcp.tool()
def server_info() -> Dict[str, Any]:
    """
    Return basic diagnostic information about this MCP server.
    """
    try:
        return fetch_server_info()
    except Exception as exc:
        log_activity(logging.ERROR, "Failed to fetch server info", error=str(exc))
        return error_response("server_info", None, exc)


# ============================================================================
# MCP Resources
# ============================================================================


@mcp.resource("file://activity.log")
def activity_log() -> str:
    """
    Expose the activity log as a readable MCP resource.
    """
    try:
        ensure_activity_log_file()
        return ACTIVITY_LOG_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        log_activity(logging.ERROR, "Failed to read activity log", error=str(exc))
        raise ActivityLogError("Unable to read activity log file.") from exc


@mcp.resource("resource://crypto/price/{symbol}")
def crypto_price_resource(symbol: str) -> str:
    """
    Expose current price as a resource.
    """
    result = get_current_price(symbol)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.resource("resource://crypto/24h/{symbol}")
def crypto_24h_resource(symbol: str) -> str:
    """
    Expose 24h ticker statistics as a resource.
    """
    result = get_24h_price_change(symbol)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.resource("resource://server/info")
def server_info_resource() -> str:
    """
    Expose local server information as a resource.
    """
    result = server_info()
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================================
# Entrypoint
# ============================================================================

if __name__ == "__main__":
    try:
        ensure_activity_log_file()
        log_activity(logging.INFO, "Starting MCP server", server_name=SERVER_NAME)
        mcp.run(transport="stdio")
    except Exception as exc:
        log_activity(logging.CRITICAL, "Fatal error while starting MCP server", error=str(exc))
        raise