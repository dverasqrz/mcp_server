"""
Binance MCP Server

Servidor MCP para consulta pública de preços e estatísticas 24h da Binance.

Ferramentas:
- get_price
- get_24h_ticker
- health_check

Requirements:
    pip install requests mcp

Execução:
    python binance_mcp.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import wraps
from time import perf_counter
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================================================================
# Configurações
# =============================================================================

SERVER_NAME = "Binance MCP"
BINANCE_API_BASE = "https://api.binance.com"
BINANCE_DATA_API_BASE = "https://data-api.binance.vision"
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_RETRY_TOTAL = 3
MAX_LOG_BODY_LENGTH = 1_000
LOG_LEVEL = os.getenv("BINANCE_MCP_LOG_LEVEL", "INFO").upper()


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(SERVER_NAME)


# =============================================================================
# MCP
# =============================================================================

mcp = FastMCP(SERVER_NAME)


# =============================================================================
# Exceções customizadas
# =============================================================================

class BinanceMCPError(Exception):
    """Exceção base do projeto."""


class InvalidInputError(BinanceMCPError):
    """Entrada inválida fornecida pelo usuário."""


class InvalidSymbolError(BinanceMCPError):
    """Símbolo inválido ou não suportado."""


class BinanceRequestError(BinanceMCPError):
    """Erro de rede/transporte ao acessar a Binance."""


class BinanceResponseError(BinanceMCPError):
    """Resposta inválida ou inesperada da Binance."""


class BinanceAPIError(BinanceMCPError):
    """Erro retornado explicitamente pela API da Binance."""


# =============================================================================
# Modelos de dados
# =============================================================================

@dataclass(frozen=True)
class PriceResult:
    symbol: str
    price: str
    source: str
    fetched_at: str


@dataclass(frozen=True)
class Ticker24hResult:
    symbol: str
    price_change: str
    price_change_percent: str
    weighted_avg_price: str
    prev_close_price: str
    last_price: str
    last_qty: str
    bid_price: str
    bid_qty: str
    ask_price: str
    ask_qty: str
    open_price: str
    high_price: str
    low_price: str
    volume: str
    quote_volume: str
    open_time: int
    close_time: int
    first_id: int
    last_id: int
    count: int
    source: str
    fetched_at: str


# =============================================================================
# Utilitários
# =============================================================================

ALIASES: dict[str, str] = {
    "btc": "BTCUSDT",
    "bitcoin": "BTCUSDT",
    "eth": "ETHUSDT",
    "ethereum": "ETHUSDT",
    "bnb": "BNBUSDT",
    "sol": "SOLUSDT",
    "solana": "SOLUSDT",
    "ada": "ADAUSDT",
    "cardano": "ADAUSDT",
    "xrp": "XRPUSDT",
    "doge": "DOGEUSDT",
    "dogecoin": "DOGEUSDT",
}


def utc_now_iso() -> str:
    """Retorna timestamp UTC em ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def truncate_text(value: str, max_length: int = MAX_LOG_BODY_LENGTH) -> str:
    """Trunca textos longos para logs."""
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}... [truncated {len(value) - max_length} chars]"


def exception_details(exc: BaseException) -> dict[str, Any]:
    """Extrai detalhes padronizados de exceções para resposta e log."""
    return {
        "exception_type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=10),
    }


def validate_decimal_string(value: str, *, field_name: str) -> None:
    """Valida string numérica retornada pela Binance."""
    try:
        Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise BinanceResponseError(
            f"Valor decimal inválido no campo '{field_name}': {value!r}"
        ) from exc


def normalize_symbol(raw_symbol: str) -> str:
    """
    Normaliza entradas amigáveis para o formato Binance.

    Exemplos aceitos:
    - btc
    - bitcoin
    - BTCUSDT
    - btc/usdt
    - btc-usdt
    - eth
    """
    if not isinstance(raw_symbol, str):
        raise InvalidInputError("O símbolo deve ser uma string.")

    cleaned = raw_symbol.strip()
    if not cleaned:
        raise InvalidInputError("O símbolo não pode ser vazio.")

    lowered = cleaned.lower()
    if lowered in ALIASES:
        return ALIASES[lowered]

    candidate = (
        cleaned.upper()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    if not candidate.isalnum():
        raise InvalidInputError(
            f"Símbolo inválido: {raw_symbol!r}. Use algo como BTC, ETH ou BTCUSDT."
        )

    known_quote_assets = ("USDT", "USDC", "BUSD", "FDUSD", "BTC", "ETH", "BNB")
    if candidate.endswith(known_quote_assets):
        return candidate

    return f"{candidate}USDT"


def success_response(data: dict[str, Any]) -> dict[str, Any]:
    """Padroniza respostas de sucesso."""
    return {
        "success": True,
        "server": SERVER_NAME,
        "timestamp": utc_now_iso(),
        "data": data,
    }


def error_response(
    *,
    error_type: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Padroniza respostas de erro para o MCP."""
    payload: dict[str, Any] = {
        "success": False,
        "server": SERVER_NAME,
        "timestamp": utc_now_iso(),
        "error": {
            "type": error_type,
            "message": message,
        },
    }
    if details:
        payload["error"]["details"] = details
    return payload


# =============================================================================
# Cliente Binance
# =============================================================================

class BinanceClient:
    """Cliente HTTP robusto para endpoints públicos da Binance."""

    def __init__(
        self,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        retry_total: int = DEFAULT_RETRY_TOTAL,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = self._build_session(retry_total=retry_total)

    @staticmethod
    def _build_session(retry_total: int) -> Session:
        session = requests.Session()

        retry = Retry(
            total=retry_total,
            connect=retry_total,
            read=retry_total,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "binance-mcp/3.0",
            }
        )
        return session

    def close(self) -> None:
        """Fecha a sessão HTTP."""
        logger.info("Closing HTTP session.")
        self.session.close()

    def get_price(self, symbol: str) -> PriceResult:
        normalized_symbol = normalize_symbol(symbol)

        payload = self._get_json(
            base_url=BINANCE_API_BASE,
            path="/api/v3/ticker/price",
            params={"symbol": normalized_symbol},
        )

        price = payload.get("price")
        response_symbol = payload.get("symbol", normalized_symbol)

        if not isinstance(response_symbol, str):
            raise BinanceResponseError(
                f"Campo 'symbol' inválido na resposta: {payload!r}"
            )

        if not isinstance(price, str):
            raise BinanceResponseError(
                f"Campo 'price' inválido na resposta: {payload!r}"
            )

        validate_decimal_string(price, field_name="price")

        return PriceResult(
            symbol=response_symbol,
            price=price,
            source="binance",
            fetched_at=utc_now_iso(),
        )

    def get_24h_ticker(self, symbol: str) -> Ticker24hResult:
        normalized_symbol = normalize_symbol(symbol)

        payload = self._get_json(
            base_url=BINANCE_DATA_API_BASE,
            path="/api/v3/ticker/24hr",
            params={"symbol": normalized_symbol},
        )

        required_types: dict[str, type] = {
            "symbol": str,
            "priceChange": str,
            "priceChangePercent": str,
            "weightedAvgPrice": str,
            "prevClosePrice": str,
            "lastPrice": str,
            "lastQty": str,
            "bidPrice": str,
            "bidQty": str,
            "askPrice": str,
            "askQty": str,
            "openPrice": str,
            "highPrice": str,
            "lowPrice": str,
            "volume": str,
            "quoteVolume": str,
            "openTime": int,
            "closeTime": int,
            "firstId": int,
            "lastId": int,
            "count": int,
        }

        for field_name, expected_type in required_types.items():
            value = payload.get(field_name)
            if not isinstance(value, expected_type):
                raise BinanceResponseError(
                    f"Campo '{field_name}' inválido. "
                    f"Esperado: {expected_type.__name__}; "
                    f"recebido: {type(value).__name__}"
                )

        decimal_fields = (
            "priceChange",
            "priceChangePercent",
            "weightedAvgPrice",
            "prevClosePrice",
            "lastPrice",
            "lastQty",
            "bidPrice",
            "bidQty",
            "askPrice",
            "askQty",
            "openPrice",
            "highPrice",
            "lowPrice",
            "volume",
            "quoteVolume",
        )

        for field_name in decimal_fields:
            validate_decimal_string(payload[field_name], field_name=field_name)

        return Ticker24hResult(
            symbol=payload["symbol"],
            price_change=payload["priceChange"],
            price_change_percent=payload["priceChangePercent"],
            weighted_avg_price=payload["weightedAvgPrice"],
            prev_close_price=payload["prevClosePrice"],
            last_price=payload["lastPrice"],
            last_qty=payload["lastQty"],
            bid_price=payload["bidPrice"],
            bid_qty=payload["bidQty"],
            ask_price=payload["askPrice"],
            ask_qty=payload["askQty"],
            open_price=payload["openPrice"],
            high_price=payload["highPrice"],
            low_price=payload["lowPrice"],
            volume=payload["volume"],
            quote_volume=payload["quoteVolume"],
            open_time=payload["openTime"],
            close_time=payload["closeTime"],
            first_id=payload["firstId"],
            last_id=payload["lastId"],
            count=payload["count"],
            source="binance",
            fetched_at=utc_now_iso(),
        )

    def health_check(self) -> dict[str, Any]:
        """Faz um teste simples de conectividade usando BTCUSDT."""
        result = self.get_price("BTCUSDT")
        return {
            "status": "ok",
            "symbol_tested": result.symbol,
            "price": result.price,
            "source": result.source,
            "fetched_at": result.fetched_at,
        }

    def _get_json(
        self,
        *,
        base_url: str,
        path: str,
        params: dict[str, str],
    ) -> dict[str, Any]:
        url = f"{base_url}{path}"
        started_at = perf_counter()

        logger.info(
            "Requesting Binance endpoint | url=%s | params=%s | timeout=%ss",
            url,
            params,
            self.timeout_seconds,
        )

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.exception(
                "Timeout while requesting Binance | url=%s | params=%s | elapsed_ms=%s",
                url,
                params,
                elapsed_ms,
            )
            raise BinanceRequestError(
                f"Timeout ao consultar a Binance em {url}"
            ) from exc
        except requests.ConnectionError as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.exception(
                "Connection error while requesting Binance | url=%s | params=%s | elapsed_ms=%s",
                url,
                params,
                elapsed_ms,
            )
            raise BinanceRequestError(
                f"Erro de conexão ao consultar a Binance em {url}"
            ) from exc
        except requests.RequestException as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.exception(
                "Generic request error while requesting Binance | url=%s | params=%s | elapsed_ms=%s",
                url,
                params,
                elapsed_ms,
            )
            raise BinanceRequestError(
                f"Erro HTTP ao consultar a Binance em {url}"
            ) from exc

        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "Received Binance response | url=%s | status=%s | reason=%s | elapsed_ms=%s | final_url=%s",
            url,
            response.status_code,
            response.reason,
            elapsed_ms,
            response.url,
        )

        return self._parse_response(response=response, url=url)

    @staticmethod
    def _parse_response(*, response: Response, url: str) -> dict[str, Any]:
        raw_text = truncate_text(response.text)

        try:
            payload = response.json()
        except ValueError as exc:
            logger.exception(
                "Non-JSON response from Binance | url=%s | status=%s | body=%s",
                url,
                response.status_code,
                raw_text,
            )
            raise BinanceResponseError(
                f"A Binance retornou resposta não-JSON em {url}"
            ) from exc

        if response.ok:
            if not isinstance(payload, dict):
                logger.error(
                    "Unexpected success payload format | url=%s | status=%s | payload_type=%s | body=%s",
                    url,
                    response.status_code,
                    type(payload).__name__,
                    raw_text,
                )
                raise BinanceResponseError(
                    f"Formato inesperado da resposta da Binance em {url}"
                )
            return payload

        if not isinstance(payload, dict):
            logger.error(
                "Unexpected error payload format | url=%s | status=%s | payload_type=%s | body=%s",
                url,
                response.status_code,
                type(payload).__name__,
                raw_text,
            )
            raise BinanceAPIError(
                f"Erro HTTP {response.status_code} com corpo inesperado em {url}"
            )

        error_code = payload.get("code")
        error_message = payload.get("msg", response.text)

        logger.error(
            "Binance API returned error | url=%s | status=%s | code=%s | msg=%s | body=%s",
            url,
            response.status_code,
            error_code,
            error_message,
            raw_text,
        )

        if error_code == -1121:
            raise InvalidSymbolError(
                f"Símbolo inválido na Binance. code={error_code}, msg={error_message}"
            )

        raise BinanceAPIError(
            f"Erro da API Binance. status={response.status_code}, "
            f"code={error_code}, msg={error_message}"
        )


client = BinanceClient()


# =============================================================================
# Decorador de segurança para tools
# =============================================================================

def safe_tool_call(func):
    """Captura exceções e retorna payload estruturado para o cliente MCP."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        started_at = perf_counter()
        logger.info(
            "Executing tool | name=%s | args=%s | kwargs=%s",
            func.__name__,
            args,
            kwargs,
        )

        try:
            result = func(*args, **kwargs)
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.info(
                "Tool executed successfully | name=%s | elapsed_ms=%s",
                func.__name__,
                elapsed_ms,
            )
            return success_response(result)

        except InvalidInputError as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.warning(
                "Invalid input in tool | name=%s | elapsed_ms=%s | error=%s",
                func.__name__,
                elapsed_ms,
                exc,
            )
            return error_response(
                error_type="InvalidInputError",
                message=str(exc),
                details=exception_details(exc),
            )

        except InvalidSymbolError as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.warning(
                "Invalid symbol in tool | name=%s | elapsed_ms=%s | error=%s",
                func.__name__,
                elapsed_ms,
                exc,
            )
            return error_response(
                error_type="InvalidSymbolError",
                message=str(exc),
                details=exception_details(exc),
            )

        except BinanceRequestError as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.error(
                "Request error in tool | name=%s | elapsed_ms=%s | error=%s",
                func.__name__,
                elapsed_ms,
                exc,
                exc_info=True,
            )
            return error_response(
                error_type="BinanceRequestError",
                message=str(exc),
                details=exception_details(exc),
            )

        except BinanceResponseError as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.error(
                "Response error in tool | name=%s | elapsed_ms=%s | error=%s",
                func.__name__,
                elapsed_ms,
                exc,
                exc_info=True,
            )
            return error_response(
                error_type="BinanceResponseError",
                message=str(exc),
                details=exception_details(exc),
            )

        except BinanceAPIError as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.error(
                "API error in tool | name=%s | elapsed_ms=%s | error=%s",
                func.__name__,
                elapsed_ms,
                exc,
                exc_info=True,
            )
            return error_response(
                error_type="BinanceAPIError",
                message=str(exc),
                details=exception_details(exc),
            )

        except Exception as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.exception(
                "Unexpected error in tool | name=%s | elapsed_ms=%s",
                func.__name__,
                elapsed_ms,
            )
            return error_response(
                error_type="UnexpectedError",
                message="Ocorreu um erro interno inesperado.",
                details=exception_details(exc),
            )

    return wrapper


# =============================================================================
# Tools MCP
# =============================================================================

@mcp.tool()
@safe_tool_call
def get_price(symbol: str) -> dict[str, Any]:
    """
    Retorna o preço atual de um ativo cripto na Binance.

    Exemplos de entrada:
    - BTC
    - bitcoin
    - BTCUSDT
    - eth
    - sol/usdt
    """
    result = client.get_price(symbol)
    return asdict(result)


@mcp.tool()
@safe_tool_call
def get_24h_ticker(symbol: str) -> dict[str, Any]:
    """
    Retorna estatísticas das últimas 24 horas de um ativo cripto na Binance.

    Inclui:
    - variação absoluta
    - variação percentual
    - máxima
    - mínima
    - volume
    - bid/ask
    """
    result = client.get_24h_ticker(symbol)
    return asdict(result)


@mcp.tool()
@safe_tool_call
def health_check() -> dict[str, Any]:
    """
    Verifica se o servidor consegue acessar a Binance corretamente.
    """
    return client.health_check()


# =============================================================================
# Execução principal
# =============================================================================

def _install_signal_handlers() -> None:
    """Registra handlers básicos para encerramento limpo."""

    def handle_signal(signum, _frame) -> None:
        logger.info("Received termination signal | signum=%s", signum)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, handle_signal)
    try:
        signal.signal(signal.SIGTERM, handle_signal)
    except AttributeError:
        logger.debug("SIGTERM handler not available on this platform.")


def main() -> int:
    """Ponto de entrada principal."""
    _install_signal_handlers()

    logger.info(
        "Starting server | name=%s | python=%s | executable=%s | log_level=%s",
        SERVER_NAME,
        sys.version.replace("\n", " "),
        sys.executable,
        LOG_LEVEL,
    )

    try:
        mcp.run(transport="stdio")
        logger.info("MCP run loop finished normally.")
        return 0

    except BrokenPipeError:
        logger.warning("STDIO pipe closed by client.")
        return 0

    except KeyboardInterrupt:
        logger.info("Server interrupted by user. Shutting down gracefully.")
        return 0

    except asyncio.CancelledError:
        logger.info("Async task cancelled during shutdown.")
        return 0

    except Exception as exc:
        logger.exception("Fatal error while running MCP server.")
        logger.error("Fatal error details: %s", exception_details(exc))
        return 1

    finally:
        client.close()
        logger.info("Server stopped.")


if __name__ == "__main__":
    sys.exit(main())