"""API key, client secret and access key families.

Each format below is taken from the vendor's own public documentation; the URL
is on the family. Only the *shape* is reproduced — the values are random, and
nothing here is ever sent anywhere to see whether it works (EXECUTION.md §8).
"""

from __future__ import annotations

import random

from datagen.families import Family, register
from datagen.families._primitives import (
    ALNUM,
    BASE62,
    UPPER,
    alnum,
    base62_crc32,
    base64_body,
    hex_lower,
    pick,
    uuid4_like,
)

LABEL = "api_key"


def _aws_access_key_id(rng: random.Random) -> str:
    """Generate an AWS access key id: ``AKIA`` plus 16 upper-case alphanumerics."""
    return "AKIA" + pick(rng, UPPER + "234567", 16)


def _aws_secret_access_key(rng: random.Random) -> str:
    """Generate an AWS secret access key: 40 base64 characters."""
    return base64_body(rng, 30)[:40]


def _github_pat_classic(rng: random.Random) -> str:
    """Generate a classic GitHub personal access token.

    ``ghp_`` followed by 30 base62 characters and a 6-character base62 CRC32
    checksum of those 30, for 36 characters after the prefix.
    """
    body = pick(rng, BASE62, 30)
    return "ghp_" + body + base62_crc32(body)


def _github_pat_fine_grained(rng: random.Random) -> str:
    """Generate a fine-grained GitHub personal access token.

    ``github_pat_`` followed by a 22-character identifier, an underscore, and a
    59-character secret whose final 6 characters are the base62 CRC32 checksum
    of the preceding 53.
    """
    identifier = pick(rng, BASE62, 22)
    body = pick(rng, BASE62, 53)
    return f"github_pat_{identifier}_{body}{base62_crc32(body)}"


def _github_app_installation_token(rng: random.Random) -> str:
    """Generate a GitHub App installation access token (``ghs_`` prefix)."""
    body = pick(rng, BASE62, 30)
    return "ghs_" + body + base62_crc32(body)


def _gitlab_pat(rng: random.Random) -> str:
    """Generate a GitLab personal access token: ``glpat-`` plus 20 characters."""
    return "glpat-" + pick(rng, ALNUM + "-_", 20)


def _slack_bot_token(rng: random.Random) -> str:
    """Generate a Slack bot token: ``xoxb-`` then two numeric parts and a secret."""
    return (
        f"xoxb-{rng.randint(10**11, 10**12 - 1)}-{rng.randint(10**12, 10**13 - 1)}-{alnum(rng, 24)}"
    )


def _slack_webhook_url(rng: random.Random) -> str:
    """Generate a Slack incoming-webhook URL, whose path is the secret."""
    return (
        "https://hooks.slack.com/services/"
        f"T{pick(rng, UPPER + '0123456789', 10)}/"
        f"B{pick(rng, UPPER + '0123456789', 10)}/{alnum(rng, 24)}"
    )


def _stripe_secret_key(rng: random.Random) -> str:
    """Generate a Stripe live secret key: ``sk_live_`` plus 24 characters."""
    return "sk_live_" + alnum(rng, 24)


def _stripe_restricted_key(rng: random.Random) -> str:
    """Generate a Stripe restricted key: ``rk_live_`` plus 24 characters."""
    return "rk_live_" + alnum(rng, 24)


def _sendgrid_api_key(rng: random.Random) -> str:
    """Generate a SendGrid API key: ``SG.`` id ``.`` secret."""
    return f"SG.{pick(rng, ALNUM + '-_', 22)}.{pick(rng, ALNUM + '-_', 43)}"


def _google_api_key(rng: random.Random) -> str:
    """Generate a Google API key: ``AIza`` plus 35 characters."""
    return "AIza" + pick(rng, ALNUM + "-_", 35)


def _openai_api_key(rng: random.Random) -> str:
    """Generate an OpenAI project API key: ``sk-proj-`` plus a long secret."""
    return "sk-proj-" + pick(rng, ALNUM + "-_", 64)


def _anthropic_api_key(rng: random.Random) -> str:
    """Generate an Anthropic API key: ``sk-ant-api03-`` plus a long secret."""
    return "sk-ant-api03-" + pick(rng, ALNUM + "-_", 95)


def _azure_storage_key(rng: random.Random) -> str:
    """Generate an Azure Storage account key: 88 base64 characters ending ``==``."""
    return base64_body(rng, 64)[:86] + "=="


def _azure_ad_client_secret(rng: random.Random) -> str:
    """Generate a Microsoft Entra ID application client secret."""
    return pick(rng, ALNUM + "-._~", 3) + "8Q~" + pick(rng, ALNUM + "-._~", 34)


def _npm_token(rng: random.Random) -> str:
    """Generate an npm automation token: ``npm_`` plus 36 characters."""
    return "npm_" + alnum(rng, 36)


def _twilio_api_key(rng: random.Random) -> str:
    """Generate a Twilio API key SID: ``SK`` plus 32 hex characters."""
    return "SK" + hex_lower(rng, 32)


def _twilio_auth_token(rng: random.Random) -> str:
    """Generate a Twilio auth token: 32 hex characters."""
    return hex_lower(rng, 32)


def _digitalocean_pat(rng: random.Random) -> str:
    """Generate a DigitalOcean personal access token: ``dop_v1_`` plus 64 hex."""
    return "dop_v1_" + hex_lower(rng, 64)


def _hashicorp_vault_token(rng: random.Random) -> str:
    """Generate a HashiCorp Vault service token: ``hvs.`` plus a long secret."""
    return "hvs." + pick(rng, ALNUM + "-_", 72)


def _pypi_token(rng: random.Random) -> str:
    """Generate a PyPI upload token: ``pypi-`` plus a base64 macaroon body."""
    return "pypi-" + base64_body(rng, 60, url_safe=True)


def _nuget_api_key(rng: random.Random) -> str:
    """Generate a NuGet API key: ``oy2`` plus 43 lower-case alphanumerics."""
    return "oy2" + pick(rng, "abcdefghijklmnopqrstuvwxyz0123456789", 43)


def _datadog_api_key(rng: random.Random) -> str:
    """Generate a Datadog API key: 32 hex characters."""
    return hex_lower(rng, 32)


def _generic_client_secret(rng: random.Random) -> str:
    """Generate an OAuth client secret with no vendor-specific shape."""
    return rng.choice([base64_body(rng, 32), hex_lower(rng, 64), uuid4_like(rng)])


def _generic_high_entropy_key(rng: random.Random) -> str:
    """Generate a hand-rolled high-entropy key of the kind internal services use."""
    length = rng.choice([32, 40, 48, 64])
    return pick(rng, ALNUM, length)


register(
    Family(
        name="aws.access_key_id",
        label=LABEL,
        generate=_aws_access_key_id,
        doc_url="https://docs.aws.amazon.com/IAM/latest/UserGuide/reference-arns.html#identifiers-unique-ids",
        note="AKIA prefix plus 16 upper-case alphanumerics.",
    )
)
register(
    Family(
        name="aws.secret_access_key",
        label=LABEL,
        generate=_aws_secret_access_key,
        doc_url="https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html",
        note="40 base64 characters with no prefix: shape alone is weak evidence, the key name carries it.",
    )
)
register(
    Family(
        name="github.pat_classic",
        label=LABEL,
        generate=_github_pat_classic,
        doc_url="https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/",
        note="ghp_ plus 30 base62 characters and a 6-character base62 CRC32 checksum.",
    )
)
register(
    Family(
        name="github.pat_fine_grained",
        label=LABEL,
        generate=_github_pat_fine_grained,
        doc_url="https://github.blog/security/application-security/introducing-fine-grained-personal-access-tokens-for-github/",
        note="github_pat_ plus a 22-character id, an underscore, and a 59-character checksummed secret.",
    )
)
register(
    Family(
        name="github.app_installation_token",
        label=LABEL,
        generate=_github_app_installation_token,
        doc_url="https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/",
        note="ghs_ prefix, same checksummed body as a classic PAT.",
    )
)
register(
    Family(
        name="gitlab.personal_access_token",
        label=LABEL,
        generate=_gitlab_pat,
        doc_url="https://docs.gitlab.com/security/tokens/",
        note="glpat- plus 20 characters.",
    )
)
register(
    Family(
        name="slack.bot_token",
        label=LABEL,
        generate=_slack_bot_token,
        doc_url="https://api.slack.com/authentication/token-types",
        note="xoxb- plus two numeric segments and a secret segment.",
    )
)
register(
    Family(
        name="slack.webhook_url",
        label=LABEL,
        generate=_slack_webhook_url,
        doc_url="https://api.slack.com/messaging/webhooks",
        note="A URL whose path is the credential: the host looks harmless, the path is not.",
    )
)
register(
    Family(
        name="stripe.secret_key",
        label=LABEL,
        generate=_stripe_secret_key,
        doc_url="https://docs.stripe.com/keys",
        note="sk_live_ plus 24 characters.",
    )
)
register(
    Family(
        name="stripe.restricted_key",
        label=LABEL,
        generate=_stripe_restricted_key,
        doc_url="https://docs.stripe.com/keys#limit-access",
        note="rk_live_ plus 24 characters.",
    )
)
register(
    Family(
        name="sendgrid.api_key",
        label=LABEL,
        generate=_sendgrid_api_key,
        doc_url="https://www.twilio.com/docs/sendgrid/ui/account-and-settings/api-keys",
        note="Three dot-separated parts: SG, a 22-character id, a 43-character secret.",
    )
)
register(
    Family(
        name="google.api_key",
        label=LABEL,
        generate=_google_api_key,
        doc_url="https://cloud.google.com/docs/authentication/api-keys",
        note="AIza plus 35 characters.",
    )
)
register(
    Family(
        name="openai.api_key",
        label=LABEL,
        generate=_openai_api_key,
        doc_url="https://platform.openai.com/docs/api-reference/authentication",
        note="sk-proj- plus a long secret.",
    )
)
register(
    Family(
        name="anthropic.api_key",
        label=LABEL,
        generate=_anthropic_api_key,
        doc_url="https://docs.claude.com/en/api/getting-started",
        note="sk-ant-api03- plus a long secret.",
    )
)
register(
    Family(
        name="azure.storage_account_key",
        label=LABEL,
        generate=_azure_storage_key,
        doc_url="https://learn.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage",
        note="88 base64 characters ending in '==': the padding is a usable signal.",
    )
)
register(
    Family(
        name="azure.ad_client_secret",
        label=LABEL,
        generate=_azure_ad_client_secret,
        doc_url="https://learn.microsoft.com/en-us/entra/identity-platform/quickstart-register-app",
        note="Contains the distinctive '8Q~' sequence seen in Entra ID client secrets.",
    )
)
register(
    Family(
        name="npm.access_token",
        label=LABEL,
        generate=_npm_token,
        doc_url="https://docs.npmjs.com/about-access-tokens",
        note="npm_ plus 36 characters.",
    )
)
register(
    Family(
        name="twilio.api_key_sid",
        label=LABEL,
        generate=_twilio_api_key,
        doc_url="https://www.twilio.com/docs/iam/api-keys",
        note="SK plus 32 hex characters.",
    )
)
register(
    Family(
        name="twilio.auth_token",
        label=LABEL,
        generate=_twilio_auth_token,
        doc_url="https://www.twilio.com/docs/iam/api-keys",
        note="Bare 32 hex characters: indistinguishable from an MD5 digest without the key name.",
    )
)
register(
    Family(
        name="digitalocean.personal_access_token",
        label=LABEL,
        generate=_digitalocean_pat,
        doc_url="https://docs.digitalocean.com/reference/api/create-personal-access-token/",
        note="dop_v1_ plus 64 hex characters.",
    )
)
register(
    Family(
        name="hashicorp.vault_service_token",
        label=LABEL,
        generate=_hashicorp_vault_token,
        doc_url="https://developer.hashicorp.com/vault/docs/concepts/tokens",
        note="hvs. plus a long secret.",
    )
)
register(
    Family(
        name="pypi.upload_token",
        label=LABEL,
        generate=_pypi_token,
        doc_url="https://pypi.org/help/#apitoken",
        note="pypi- plus a base64 macaroon.",
    )
)
register(
    Family(
        name="nuget.api_key",
        label=LABEL,
        generate=_nuget_api_key,
        doc_url="https://learn.microsoft.com/en-us/nuget/nuget-org/publish-a-package",
        note="oy2 plus 43 lower-case alphanumerics.",
    )
)
register(
    Family(
        name="datadog.api_key",
        label=LABEL,
        generate=_datadog_api_key,
        doc_url="https://docs.datadoghq.com/account_management/api-app-keys/",
        note="Bare 32 hex characters, like the Twilio auth token.",
    )
)
register(
    Family(
        name="generic.oauth_client_secret",
        label=LABEL,
        generate=_generic_client_secret,
        doc_url="No vendor format: OAuth 2.0 (RFC 6749) does not prescribe a client-secret shape, so deployments use base64, hex or a UUID.",
        note="Covers the common hand-rolled shapes.",
    )
)
register(
    Family(
        name="generic.high_entropy_key",
        label=LABEL,
        generate=_generic_high_entropy_key,
        doc_url="No vendor format: this is the hand-rolled internal key that entropy heuristics must catch.",
        note="32 to 64 mixed-case alphanumerics with no prefix at all.",
    )
)
