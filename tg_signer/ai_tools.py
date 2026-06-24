import asyncio
import base64
import io
import json
import logging
import os
import pathlib
import re
from typing import TYPE_CHECKING, Any, Union

import httpx
import json_repair
from typing_extensions import Optional, Required, TypedDict

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow is optional at runtime
    Image = None

if TYPE_CHECKING:
    from openai import AsyncOpenAI  # 在性能弱的机器上导入openai包实在有些慢

from tg_signer.utils import UserInput, print_to_user

DEFAULT_MODEL = "gpt-4o"
DEFAULT_PADDLEOCR_API_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_PADDLEOCR_MODEL = "PaddleOCR-VL-1.5"

DEFAULT_CHOOSE_OPTION_BY_IMAGE_PROMPT = (
    "You are a low-latency visual matcher for Telegram sign-in challenges. "
    "Choose exactly one option whose text best matches the main object or "
    "concept shown in the image and the question. Ignore retry warnings, "
    "time-limit reminders, and other unrelated footer text. Return JSON only: "
    '{"option":1}. The option value must be one of the provided indexes, '
    "starting at 1. If JSON mode is unavailable, return only the chosen option "
    "index or option text."
)

DEFAULT_CHOOSE_OPTIONS_BY_IMAGE_PROMPT = (
    "You solve Telegram bot image or text challenges. Read only the actual "
    "question and the button list. Use the image when the question refers to "
    "the picture. Unless the question explicitly asks for multiple clicks or "
    "building a phrase, return exactly one option. Ignore retry warnings, "
    "time-limit reminders, and unrelated footer text. Return JSON only: "
    '{"options":[1]}. '
    "The options field must be a list of option indexes starting at 1. "
    "If only one click is needed, return a one-item list. If JSON mode is "
    "unavailable, return only the chosen option index or option text."
)

DEFAULT_SINGLE_OBJECT_CHOICE_PROMPT = (
    "You are a fast image classifier for a Telegram sign-in button challenge. "
    "The image usually contains one main object on a clean background. Pick "
    "the single button whose text best names that object. Do not explain. "
    'Return JSON only: {"options":[1]}. '
    "The option indexes start at 1. If JSON mode is unavailable, return only "
    "the chosen option index or option text."
)

DEFAULT_EXTRACT_TEXT_BY_IMAGE_PROMPT = (
    "You are an OCR assistant. Extract the most relevant text from the image. "
    "Return plain text only, no markdown, no explanation."
)

DEFAULT_CALCULATE_PROBLEM_PROMPT = (
    "你是一个**答题助手**，可以根据用户的问题给出正确的回答，只需要回复答案，不要解释，不要输出任何其他内容。"
)


def encode_image(image: bytes):
    return base64.b64encode(image).decode("utf-8")


logger = logging.getLogger("tg-signer")


class OpenAIConfig(TypedDict, total=False):
    api_key: Required[str]
    base_url: Optional[str]
    model: Optional[str]


class OpenAIConfigManager:
    def __init__(self, workdir: Union[str, pathlib.Path]):
        self.workdir = pathlib.Path(workdir)

    def get_config_file(self) -> pathlib.Path:
        return self.workdir / ".openai_config.json"

    def has_env_config(self):
        return bool(
            os.environ.get("OPENAI_API_KEY")
            or (
                os.environ.get("PADDLEOCR_API_TOKEN")
                and (
                    os.environ.get("PADDLEOCR_API_URL")
                    or DEFAULT_PADDLEOCR_API_URL
                )
            )
        )

    def has_config(self) -> bool:
        return bool(self.load_config())

    def load_file_config(self) -> Optional[dict]:
        config_file = self.get_config_file()
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as fp:
                c = json.load(fp)
            # 简单验证必需字段
            if "api_key" in c:
                return c
        return None

    def save_config(self, api_key: str, base_url: str = None, model: str = None):
        config_file = self.get_config_file()
        config = OpenAIConfig(api_key=api_key, base_url=base_url, model=model)
        with open(config_file, "w", encoding="utf-8") as fp:
            json.dump(config, fp, ensure_ascii=False, indent=2)

    def load_config(self) -> Optional[OpenAIConfig]:
        # 环境变量优先
        if self.has_env_config():
            if not os.environ.get("OPENAI_API_KEY"):
                return OpenAIConfig(api_key="")
            return OpenAIConfig(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
                model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
            )
        return self.load_file_config()

    def ask_for_config(self):
        print_to_user("开始配置OpenAI API并保存至本地。")
        input_ = UserInput()
        api_key = input_("请输入 OPENAI_API_KEY: ").strip()
        while not api_key:
            print_to_user("API Key不能为空！")
            api_key = input_("请输入 OPENAI_API_KEY: ").strip()

        base_url = (
            input_(
                "请输入 OPENAI_BASE_URL (可选，直接回车使用默认OpenAI地址): "
            ).strip()
            or None
        )
        model = (
            input_(
                f"请输入 OPENAI_MODEL (可选，直接回车使用默认模型({DEFAULT_MODEL})): "
            ).strip()
            or None
        )
        self.save_config(api_key, base_url=base_url, model=model)
        print_to_user("OpenAI配置已保存。")
        return self.load_config()


def get_openai_client(
    api_key: str = None,
    base_url: str = None,
    **kwargs,
) -> Optional["AsyncOpenAI"]:
    from openai import AsyncOpenAI, OpenAIError

    try:
        return AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)
    except OpenAIError:
        return None


class AITools:
    _QUESTION_LINE_HINTS = (
        "点击",
        "选择",
        "选出",
        "找出",
        "识别",
        "图中",
        "图片",
        "图里",
        "图上的",
        "图示",
        "image",
        "photo",
        "picture",
        "shown",
        "select",
        "choose",
        "click",
    )

    def __init__(self, cfg: OpenAIConfig):
        self.client = None
        if cfg.get("api_key"):
            self.client = get_openai_client(
                api_key=cfg["api_key"], base_url=cfg.get("base_url")
            )
        self.default_model = cfg.get("model") or DEFAULT_MODEL

    @staticmethod
    def _normalize_option_text(text: Any) -> str:
        return "".join(str(text).split()).lower()

    @staticmethod
    def _has_paddleocr_config() -> bool:
        return bool(os.environ.get("PADDLEOCR_API_TOKEN"))

    @classmethod
    def _require_paddleocr_config(cls) -> None:
        if not cls._has_paddleocr_config():
            raise RuntimeError(
                "PADDLEOCR_API_TOKEN is required for image AI actions; "
                "OpenAI/OpenRouter/Gemini fallback is disabled"
            )

    @staticmethod
    def _paddleocr_api_url() -> str:
        return os.environ.get("PADDLEOCR_API_URL") or DEFAULT_PADDLEOCR_API_URL

    @staticmethod
    def _paddleocr_model(kind: str = "default") -> str:
        if kind == "text":
            return (
                os.environ.get("PADDLEOCR_TEXT_MODEL")
                or os.environ.get("PADDLEOCR_MODEL")
                or "PP-OCRv5"
            )
        if kind == "choice":
            return (
                os.environ.get("PADDLEOCR_CHOICE_MODEL")
                or os.environ.get("PADDLEOCR_MODEL")
                or DEFAULT_PADDLEOCR_MODEL
            )
        return os.environ.get("PADDLEOCR_MODEL") or DEFAULT_PADDLEOCR_MODEL

    @staticmethod
    def _paddleocr_timeout() -> float:
        try:
            timeout = float(os.environ.get("PADDLEOCR_TIMEOUT", "18"))
        except ValueError:
            return 18.0
        return max(3.0, timeout)

    @staticmethod
    def _paddleocr_poll_interval() -> float:
        try:
            interval = float(os.environ.get("PADDLEOCR_POLL_INTERVAL", "0.8"))
        except ValueError:
            return 0.8
        return max(0.2, interval)

    @staticmethod
    def _ai_timeout() -> float:
        try:
            timeout = float(os.environ.get("AI_VISION_TIMEOUT", "8"))
        except ValueError:
            return 8.0
        return max(3.0, timeout)

    @staticmethod
    def _read_positive_int_env(name: str, default: int, minimum: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default
        return max(minimum, value)

    @classmethod
    def _extract_relevant_query(cls, query: str) -> str:
        if not query:
            return ""
        lines = []
        for raw_line in str(query).splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if line:
                lines.append(line)
        if not lines:
            return ""

        for line in lines:
            lowered = line.lower()
            if any(hint in line or hint in lowered for hint in cls._QUESTION_LINE_HINTS):
                return line[:160]
        return lines[0][:160]

    @classmethod
    def _looks_like_single_object_choice(
        cls, query: str, options: list[tuple[int, str]]
    ) -> bool:
        if not options or len(options) > 8:
            return False
        normalized_query = cls._extract_relevant_query(query).lower()
        if not any(
            keyword in normalized_query
            for keyword in ("图", "图片", "image", "photo", "picture", "object")
        ):
            return False
        short_label_count = sum(
            1
            for _, option_text in options
            if 0 < len(cls._normalize_option_text(option_text)) <= 16
        )
        return short_label_count == len(options)

    @classmethod
    def _crop_light_border(cls, image: "Image.Image") -> "Image.Image":
        white_threshold = cls._read_positive_int_env(
            "AI_VISION_WHITE_THRESHOLD", 245, 200
        )
        mask = image.convert("L").point(lambda px: 255 if px < white_threshold else 0)
        bbox = mask.getbbox()
        if not bbox or bbox == (0, 0, image.width, image.height):
            return image

        padding = max(12, min(image.size) // 32)
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(image.width, bbox[2] + padding)
        bottom = min(image.height, bbox[3] + padding)
        return image.crop((left, top, right, bottom))

    @classmethod
    def _prepare_vision_image(cls, image: bytes) -> bytes:
        if Image is None:
            return image

        try:
            with Image.open(io.BytesIO(image)) as raw_image:
                prepared = raw_image.convert("RGB")
        except Exception:
            return image

        prepared = cls._crop_light_border(prepared)
        max_edge = cls._read_positive_int_env("AI_VISION_MAX_EDGE", 640, 224)
        if max(prepared.size) > max_edge:
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            prepared.thumbnail((max_edge, max_edge), resampling)

        quality = cls._read_positive_int_env("AI_VISION_JPEG_QUALITY", 85, 40)
        output = io.BytesIO()
        prepared.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()

    @classmethod
    def _prepare_ocr_image(cls, image: bytes) -> bytes:
        if Image is None:
            return image

        try:
            with Image.open(io.BytesIO(image)) as raw_image:
                prepared = raw_image.convert("RGB")
        except Exception:
            return image

        prepared = cls._crop_light_border(prepared)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        scale = min(
            cls._read_positive_int_env("PADDLEOCR_TEXT_IMAGE_SCALE", 3, 1),
            6,
        )
        max_edge = cls._read_positive_int_env("PADDLEOCR_TEXT_MAX_EDGE", 1600, 480)
        if scale > 1 and max(prepared.size) * scale <= max_edge:
            prepared = prepared.resize(
                (prepared.width * scale, prepared.height * scale),
                resampling,
            )
        elif max(prepared.size) > max_edge:
            prepared.thumbnail((max_edge, max_edge), resampling)

        output = io.BytesIO()
        prepared.save(output, format="PNG", optimize=True)
        return output.getvalue()

    @staticmethod
    def _format_option_lines(options: list[tuple[int, str]]) -> str:
        return "\n".join(f"{index}. {text}" for index, text in options)

    @classmethod
    def _coerce_option_index(cls, result: Any, options: list[tuple[int, str]]) -> int:
        if isinstance(result, list):
            result = next((item for item in result if item is not None), None)

        if isinstance(result, dict):
            if isinstance(result.get("options"), list) and result["options"]:
                result = result["options"][0]
            else:
                for key in ("option", "index", "choice", "answer", "button", "text"):
                    if key in result:
                        result = result[key]
                        break

        if isinstance(result, dict):
            raise ValueError(f"AI result does not contain an option: {result}")

        if isinstance(result, int):
            return result

        if isinstance(result, str):
            stripped = result.strip()
            if stripped.lstrip("+-").isdigit():
                return int(stripped)
            normalized_result = cls._normalize_option_text(stripped)
            for index, option_text in options:
                normalized_option = cls._normalize_option_text(option_text)
                if normalized_result == normalized_option:
                    return index
            for index, option_text in options:
                normalized_option = cls._normalize_option_text(option_text)
                if normalized_option and normalized_option in normalized_result:
                    return index

        raise ValueError(f"Could not parse AI option result: {result}")

    @classmethod
    def _coerce_option_indexes(cls, result: Any, options: list[tuple[int, str]]) -> list[int]:
        if isinstance(result, list):
            if len(result) == 1 and isinstance(result[0], dict):
                result = result[0]
            else:
                return [cls._coerce_option_index(item, options) for item in result]

        if isinstance(result, dict):
            raw_options = result.get("options")
            if raw_options is None:
                raw_options = result.get("option")
            if raw_options is not None:
                if not isinstance(raw_options, list):
                    raw_options = [raw_options]
                return [cls._coerce_option_index(item, options) for item in raw_options]

        return [cls._coerce_option_index(result, options)]

    @classmethod
    def _collect_paddleocr_texts(cls, node: Any) -> list[str]:
        texts: list[str] = []
        if isinstance(node, list):
            for item in node:
                texts.extend(cls._collect_paddleocr_texts(item))
            return texts

        if not isinstance(node, dict):
            return texts

        markdown = node.get("markdown")
        if isinstance(markdown, dict) and isinstance(markdown.get("text"), str):
            text = markdown["text"].strip()
            if text:
                texts.append(text)

        pruned_result = node.get("prunedResult")
        if isinstance(pruned_result, dict):
            texts.extend(cls._collect_paddleocr_texts(pruned_result))

        rec_texts = node.get("rec_texts")
        if isinstance(rec_texts, list):
            text = " ".join(str(item).strip() for item in rec_texts if str(item).strip())
            if text:
                texts.append(text)

        for key in ("text", "content", "value", "recognized_text", "ocr_text"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())

        for key in (
            "result",
            "data",
            "layoutParsingResults",
            "ocrResults",
            "extractResult",
        ):
            value = node.get(key)
            if isinstance(value, (dict, list)):
                texts.extend(cls._collect_paddleocr_texts(value))

        return texts

    @classmethod
    def _extract_paddleocr_text(cls, response: Any) -> str:
        seen = set()
        unique_texts = []
        for text in cls._collect_paddleocr_texts(response):
            normalized = text.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_texts.append(normalized)
        return "\n".join(unique_texts).strip()

    @classmethod
    def _coerce_paddleocr_option_indexes(
        cls, response: Any, options: list[tuple[int, str]]
    ) -> list[int]:
        text = cls._extract_paddleocr_text(response)
        if not text:
            raise ValueError("PaddleOCR returned no text for option matching")

        parsed: Any = text
        if any(marker in text for marker in ("{", "[", '"options"', '"option"')):
            try:
                parsed = json_repair.loads(text)
            except Exception:
                parsed = text
        return cls._coerce_option_indexes(parsed, options)

    @staticmethod
    def _should_retry_without_json_mode(exc: Exception) -> bool:
        text = str(exc).lower()
        indicators = (
            "response_format",
            "json_object",
            "bad_response_status_code",
            "openai_error",
            "unsupported",
        )
        return any(indicator in text for indicator in indicators)

    @staticmethod
    def _get_exception_status_code(exc: Exception) -> int | None:
        for attr in ("status_code", "code"):
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)

        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value

        text = str(exc)
        for pattern in (r"Error code:\s*(\d{3})", r"['\"]code['\"]:\s*(\d{3})"):
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        return None

    @classmethod
    def _should_retry_transient_ai_error(cls, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True

        text = str(exc).lower()
        quota_markers = (
            "quota exceeded",
            "resource_exhausted",
            "free_tier",
            "check your plan and billing",
        )
        if any(marker in text for marker in quota_markers):
            return False

        status_code = cls._get_exception_status_code(exc)
        if status_code in {429, 500, 502, 503, 504}:
            return True

        transient_markers = (
            "unavailable",
            "high demand",
            "rate limit",
            "rate_limit",
            "temporarily unavailable",
            "try again later",
            "server error",
            "bad gateway",
            "gateway timeout",
        )
        return any(marker in text for marker in transient_markers)

    @classmethod
    def _vision_retry_attempts(cls) -> int:
        return cls._read_positive_int_env("AI_VISION_RETRY_ATTEMPTS", 2, 1)

    @staticmethod
    def _vision_retry_delay(attempt: int) -> float:
        try:
            base_delay = float(os.environ.get("AI_VISION_RETRY_DELAY", "0.6"))
        except ValueError:
            base_delay = 0.6
        return max(0.0, base_delay) * attempt

    async def _call_visual_completion_with_retries(self, client: "AsyncOpenAI", kwargs):
        attempts = self._vision_retry_attempts()
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=self._ai_timeout(),
                )
            except Exception as exc:
                last_error = exc
                if (
                    attempt >= attempts
                    or not self._should_retry_transient_ai_error(exc)
                ):
                    raise
                delay = self._vision_retry_delay(attempt)
                logger.warning(
                    "Transient AI provider error, retrying visual request "
                    "(attempt %s/%s, delay %.1fs): %s",
                    attempt + 1,
                    attempts,
                    delay,
                    exc,
                )
                if delay:
                    await asyncio.sleep(delay)
        raise last_error or RuntimeError("AI visual request failed")

    async def _create_visual_completion(
        self,
        *,
        client: "AsyncOpenAI",
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        expect_json: bool,
    ):
        kwargs = {
            "messages": messages,
            "model": model,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if expect_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            return await self._call_visual_completion_with_retries(client, kwargs)
        except Exception as exc:
            if not expect_json or not self._should_retry_without_json_mode(exc):
                raise
            logger.warning(
                "AI provider rejected structured JSON mode, retrying without response_format: %s",
                exc,
            )
            kwargs.pop("response_format", None)
            return await self._call_visual_completion_with_retries(client, kwargs)

    @classmethod
    async def _fetch_paddleocr_jsonl(
        cls, client: httpx.AsyncClient, jsonl_url: str
    ) -> list[dict[str, Any]]:
        response = await client.get(jsonl_url)
        response.raise_for_status()
        results = []
        for line in response.text.splitlines():
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
        return results

    @classmethod
    def _paddleocr_optional_payload(cls, model: str) -> dict[str, Any]:
        optional_payload: dict[str, Any] = {
            "useDocOrientationClassify": False,
        }
        if model.startswith("PaddleOCR-VL"):
            optional_payload.update(
                {
                    "useDocUnwarping": False,
                    "useChartRecognition": False,
                }
            )
        return optional_payload

    async def _request_paddleocr(
        self, image: bytes, *, model: str | None = None, kind: str = "choice"
    ) -> dict[str, Any]:
        token = os.environ.get("PADDLEOCR_API_TOKEN")
        if not token:
            raise RuntimeError("PADDLEOCR_API_TOKEN is not configured")

        if kind == "text":
            image = self._prepare_ocr_image(image)
            filename = "image.png"
            content_type = "image/png"
        else:
            image = self._prepare_vision_image(image)
            filename = "image.jpg"
            content_type = "image/jpeg"
        model = model or self._paddleocr_model()
        timeout_seconds = self._paddleocr_timeout()
        timeout = httpx.Timeout(timeout_seconds, connect=5.0)
        headers = {"Authorization": f"Bearer {token}"}
        optional_payload = self._paddleocr_optional_payload(model)
        data = {
            "model": model,
            "optionalPayload": json.dumps(optional_payload),
        }
        files = {"file": (filename, image, content_type)}
        api_url = self._paddleocr_api_url().rstrip("/")
        deadline = asyncio.get_running_loop().time() + timeout_seconds

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                api_url, headers=headers, data=data, files=files
            )
            response.raise_for_status()
            payload = response.json()
            job_id = (payload.get("data") or {}).get("jobId")
            if not job_id:
                return payload

            while True:
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("PaddleOCR job timed out")

                job_response = await client.get(f"{api_url}/{job_id}", headers=headers)
                job_response.raise_for_status()
                job_payload = job_response.json()
                job_data = job_payload.get("data") or {}
                state = job_data.get("state")
                if state == "done":
                    result_url = (job_data.get("resultUrl") or {}).get("jsonUrl")
                    if result_url:
                        job_data["result"] = await self._fetch_paddleocr_jsonl(
                            client, result_url
                        )
                    return job_payload
                if state == "failed":
                    raise RuntimeError(
                        f"PaddleOCR job failed: {job_data.get('errorMsg') or job_payload}"
                    )
                if state not in {"pending", "running", None}:
                    raise RuntimeError(f"Unexpected PaddleOCR job state: {state}")

                delay = min(
                    self._paddleocr_poll_interval(),
                    max(0.0, deadline - asyncio.get_running_loop().time()),
                )
                if delay:
                    await asyncio.sleep(delay)

    async def choose_options_by_paddleocr(
        self, image: bytes, options: list[tuple[int, str]]
    ) -> list[int]:
        response = await self._request_paddleocr(
            image, model=self._paddleocr_model("choice"), kind="choice"
        )
        return self._coerce_paddleocr_option_indexes(response, options)

    async def extract_text_by_paddleocr(self, image: bytes) -> str:
        response = await self._request_paddleocr(
            image, model=self._paddleocr_model("text"), kind="text"
        )
        text = self._extract_paddleocr_text(response)
        if text:
            logger.info("PaddleOCR text preview: %s", text.replace("\n", " ")[:120])
        return text

    async def choose_option_by_image(
        self,
        image: bytes,
        query: str,
        options: list[tuple[int, str]],
        client: "AsyncOpenAI" = None,
        model: str = None,
        system_prompt: str | None = None,
        temperature=0.1,
    ) -> int:
        self._require_paddleocr_config()
        return (await self.choose_options_by_paddleocr(image, options))[0]

    async def choose_options_by_image(
        self,
        image: bytes,
        query: str,
        options: list[tuple[int, str]],
        client: "AsyncOpenAI" = None,
        model: str = None,
        system_prompt: str | None = None,
        temperature=0.1,
    ) -> list[int]:
        self._require_paddleocr_config()
        return await self.choose_options_by_paddleocr(image, options)

    async def extract_text_by_image(
        self,
        image: bytes,
        query: str = "",
        client: "AsyncOpenAI" = None,
        model: str = None,
        system_prompt: str | None = None,
        temperature=0.1,
    ) -> str:
        self._require_paddleocr_config()
        return await self.extract_text_by_paddleocr(image)

    async def calculate_problem(
        self,
        query: str,
        client: "AsyncOpenAI" = None,
        model: str = None,
        system_prompt: str | None = None,
        temperature=0.1,
    ) -> str:
        sys_prompt = (system_prompt or "").strip() or DEFAULT_CALCULATE_PROBLEM_PROMPT
        model = model or self.default_model
        client = client or self.client
        text = f"问题是: {query}\n\n只需要给出答案，不要解释，不要输出任何其他内容。The answer is:"
        # noinspection PyTypeChecker
        completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text},
            ],
            model=model,
            stream=False,
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()

    async def get_reply(
        self,
        prompt: str,
        query: str,
        client: "AsyncOpenAI" = None,
        model: str = None,
    ) -> str:
        model = model or self.default_model
        client = client or self.client
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": f"{query}"},
        ]
        # noinspection PyTypeChecker
        completion = await client.chat.completions.create(
            messages=messages,
            model=model,
            stream=False,
        )
        message = completion.choices[0].message
        return message.content
