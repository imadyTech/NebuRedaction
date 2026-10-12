"""Connection string and embedded-credential URL families.

These are the values where the credential is one field inside a longer string.
Masking the whole value is usually right, and plan §3.2's fingerprint scheme
handles it, but the model still has to recognise the container — which is why
the non-secret parts (host names, database names, options) are generated too
rather than being fixed filler.
"""

from __future__ import annotations

import random

from datagen.families import Family, register
from datagen.families._primitives import hostname, url_safe_password, username

LABEL = "connection_string"

_DATABASES = ("orders", "payments", "inventory", "nebulog", "reporting", "appdb")


def _mssql_connection_string(rng: random.Random) -> str:
    """Generate an ADO.NET SQL Server connection string with a password."""
    parts = [
        f"Server={hostname(rng)},1433",
        f"Database={rng.choice(_DATABASES)}",
        f"User Id={username(rng)}",
        f"Password={url_safe_password(rng)}",
    ]
    if rng.random() < 0.5:
        parts.append("Encrypt=True")
    if rng.random() < 0.3:
        parts.append("TrustServerCertificate=False")
    return ";".join(parts) + ";"


def _postgres_url(rng: random.Random) -> str:
    """Generate a PostgreSQL connection URI."""
    scheme = rng.choice(["postgresql", "postgres"])
    tail = "?sslmode=require" if rng.random() < 0.5 else ""
    return (
        f"{scheme}://{username(rng)}:{url_safe_password(rng)}@"
        f"{hostname(rng)}:5432/{rng.choice(_DATABASES)}{tail}"
    )


def _mysql_url(rng: random.Random) -> str:
    """Generate a MySQL connection URI."""
    return (
        f"mysql://{username(rng)}:{url_safe_password(rng)}@"
        f"{hostname(rng)}:3306/{rng.choice(_DATABASES)}"
    )


def _mongodb_srv_url(rng: random.Random) -> str:
    """Generate a MongoDB SRV connection string."""
    return (
        f"mongodb+srv://{username(rng)}:{url_safe_password(rng)}@"
        f"{hostname(rng)}/{rng.choice(_DATABASES)}?retryWrites=true&w=majority"
    )


def _redis_url(rng: random.Random) -> str:
    """Generate a Redis URL, which carries a password and often no user name."""
    user = username(rng) if rng.random() < 0.3 else ""
    return f"redis://{user}:{url_safe_password(rng)}@{hostname(rng)}:6379/{rng.randint(0, 15)}"


def _amqp_url(rng: random.Random) -> str:
    """Generate an AMQP (RabbitMQ) URL."""
    return f"amqp://{username(rng)}:{url_safe_password(rng)}@{hostname(rng)}:5672/%2F"


def _azure_storage_connection_string(rng: random.Random) -> str:
    """Generate an Azure Storage connection string, which embeds the account key."""
    from datagen.families._primitives import base64_body

    account = rng.choice(["ordersstore", "nebulogdata", "paymentsblob"])
    key = base64_body(rng, 64)[:86] + "=="
    return (
        "DefaultEndpointsProtocol=https;"
        f"AccountName={account};AccountKey={key};"
        "EndpointSuffix=core.windows.net"
    )


def _http_basic_in_url(rng: random.Random) -> str:
    """Generate an HTTPS URL with userinfo credentials, as RFC 3986 permits."""
    path = rng.choice(["/api/v1/orders", "/feed.xml", "/artifacts/latest", "/health"])
    return f"https://{username(rng)}:{url_safe_password(rng)}@{hostname(rng)}{path}"


def _smtp_url(rng: random.Random) -> str:
    """Generate an SMTP URL with embedded credentials."""
    return f"smtps://{username(rng)}:{url_safe_password(rng)}@{hostname(rng)}:587"


register(
    Family(
        name="mssql.connection_string",
        label=LABEL,
        generate=_mssql_connection_string,
        doc_url="https://learn.microsoft.com/en-us/sql/connect/ado-net/connection-string-syntax",
        note="Semicolon-delimited keywords; the ';Password=' marker is the strongest signal.",
    )
)
register(
    Family(
        name="postgres.connection_uri",
        label=LABEL,
        generate=_postgres_url,
        doc_url="https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING",
        note="postgresql:// or postgres:// with userinfo credentials.",
    )
)
register(
    Family(
        name="mysql.connection_uri",
        label=LABEL,
        generate=_mysql_url,
        doc_url="https://dev.mysql.com/doc/refman/8.4/en/connecting-using-uri-or-key-value-pairs.html",
        note="mysql:// with userinfo credentials.",
    )
)
register(
    Family(
        name="mongodb.srv_connection_string",
        label=LABEL,
        generate=_mongodb_srv_url,
        doc_url="https://www.mongodb.com/docs/manual/reference/connection-string/",
        note="mongodb+srv:// with userinfo credentials and query options.",
    )
)
register(
    Family(
        name="redis.connection_uri",
        label=LABEL,
        generate=_redis_url,
        doc_url="https://www.iana.org/assignments/uri-schemes/prov/redis",
        note="Often has an empty user name, so the ':password@' sits right after '//'.",
    )
)
register(
    Family(
        name="amqp.connection_uri",
        label=LABEL,
        generate=_amqp_url,
        doc_url="https://www.rabbitmq.com/docs/uri-spec",
        note="amqp:// with userinfo credentials and a percent-encoded vhost.",
    )
)
register(
    Family(
        name="azure.storage_connection_string",
        label=LABEL,
        generate=_azure_storage_connection_string,
        doc_url="https://learn.microsoft.com/en-us/azure/storage/common/storage-configure-connection-string",
        note="Keyword form carrying AccountKey: a connection string that contains an API key.",
    )
)
register(
    Family(
        name="http.basic_credentials_in_url",
        label=LABEL,
        generate=_http_basic_in_url,
        doc_url="https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.1",
        note="The '://user:pass@' structure flagged as a feature in plan §4.2.",
    )
)
register(
    Family(
        name="smtp.connection_uri",
        label=LABEL,
        generate=_smtp_url,
        doc_url="https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.1",
        note="Mail submission URL with embedded credentials.",
    )
)
