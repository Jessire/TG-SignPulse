import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from io import BytesIO
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from PIL import Image

from tg_signer.ai_tools import AITools, OpenAIConfigManager
from tg_signer.config import ReplyByImageRecognitionAction
from tg_signer.core import UserSigner, _is_callback_confirmation_unavailable


class AIToolsOptionParsingTest(unittest.TestCase):
    def setUp(self):
        self.options = [(1, "social"), (2, "shopping"), (3, "lipstick"), (4, "mask")]

    def test_coerce_option_index_accepts_list_response(self):
        self.assertEqual(AITools._coerce_option_index([{"option": 4}], self.options), 4)

    def test_coerce_option_index_accepts_answer_text(self):
        self.assertEqual(AITools._coerce_option_index({"answer": "mask"}, self.options), 4)

    def test_coerce_option_indexes_accepts_list_payload(self):
        self.assertEqual(AITools._coerce_option_indexes([{"options": [4]}], self.options), [4])

    def test_coerce_option_indexes_accepts_text_payload(self):
        self.assertEqual(AITools._coerce_option_indexes({"answer": "mask"}, self.options), [4])

    def test_coerce_option_index_rejects_unknown_response(self):
        with self.assertRaises(ValueError):
            AITools._coerce_option_index({"reason": "no option"}, self.options)

    def test_extract_relevant_query_prefers_question_line(self):
        query = (
            "请在 30 秒内点击图中事物的按钮以完成签到\n\n"
            "每天只有一次机会, 失败或者过期当天不可重试"
        )
        self.assertEqual(
            AITools._extract_relevant_query(query),
            "请在 30 秒内点击图中事物的按钮以完成签到",
        )

    def test_prepare_vision_image_resizes_large_input(self):
        image = Image.new("RGB", (1600, 1200), "white")
        for x in range(420, 1180):
            for y in range(260, 940):
                image.putpixel((x, y), (20, 20, 20))

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        prepared = AITools._prepare_vision_image(buffer.getvalue())
        with Image.open(BytesIO(prepared)) as prepared_image:
            self.assertLessEqual(max(prepared_image.size), 640)
            self.assertLess(prepared_image.width, 1600)
            self.assertLess(prepared_image.height, 1200)

    def test_prepare_ocr_image_denoises_and_upscales_small_input(self):
        image = Image.new("RGB", (160, 50), (60, 120, 230))
        for x in range(40, 120):
            for y in range(15, 35):
                if (x + y) % 5 == 0:
                    image.putpixel((x, y), (20, 20, 30))

        buffer = BytesIO()
        image.save(buffer, format="JPEG")

        with patch.dict("os.environ", {"PADDLEOCR_TEXT_IMAGE_SCALE": "3"}):
            prepared = AITools._prepare_ocr_image(buffer.getvalue())

        with Image.open(BytesIO(prepared)) as prepared_image:
            self.assertEqual(prepared_image.format, "PNG")
            self.assertGreaterEqual(prepared_image.width, 480)
            self.assertGreaterEqual(prepared_image.height, 150)

    def test_extract_yellow_text_mask_keeps_yellow_captcha_text(self):
        image = Image.new("RGB", (80, 30), (80, 55, 150))
        for x in range(10, 30):
            for y in range(6, 24):
                image.putpixel((x, y), (230, 210, 70))

        mask = AITools._extract_yellow_text_mask(image)

        self.assertIsNotNone(mask)
        self.assertEqual(mask.getpixel((15, 10)), 0)
        self.assertEqual(mask.getpixel((60, 10)), 255)

    def test_timeout_is_treated_as_transient_ai_error(self):
        self.assertTrue(AITools._should_retry_transient_ai_error(TimeoutError()))

    def test_quota_exhaustion_is_not_retried_as_transient_error(self):
        error = RuntimeError(
            "Error code: 429 - {'error': {'status': 'RESOURCE_EXHAUSTED', "
            "'message': 'You exceeded your current quota, free_tier'}}"
        )

        self.assertFalse(AITools._should_retry_transient_ai_error(error))

    def test_extracts_text_from_paddleocr_markdown_result(self):
        response = {
            "result": {
                "layoutParsingResults": [
                    {"markdown": {"text": " bxtG\n"}},
                ]
            }
        }

        self.assertEqual(AITools._extract_paddleocr_text(response), "bxtG")

    def test_extracts_text_from_paddleocr_pruned_result(self):
        response = {
            "result": {
                "layoutParsingResults": [
                    {"prunedResult": {"rec_texts": ["b", "x", "t", "G"]}},
                ]
            }
        }

        self.assertEqual(AITools._extract_paddleocr_text(response), "b x t G")

    def test_extracts_text_from_paddleocr_jsonl_job_result(self):
        response = {
            "data": {
                "state": "done",
                "result": [
                    {
                        "result": {
                            "layoutParsingResults": [
                                {"markdown": {"text": "mask"}},
                            ]
                        }
                    }
                ],
            }
        }

        self.assertEqual(AITools._extract_paddleocr_text(response), "mask")

    def test_coerces_paddleocr_text_to_option(self):
        response = {
            "result": {
                "layoutParsingResults": [
                    {"markdown": {"text": "mask"}},
                ]
            }
        }

        self.assertEqual(
            AITools._coerce_paddleocr_option_indexes(response, self.options),
            [4],
        )

    def test_coerces_paddleocr_json_to_option(self):
        response = {
            "result": {
                "layoutParsingResults": [
                    {"markdown": {"text": '{"options":[2]}'}},
                ]
            }
        }

        self.assertEqual(
            AITools._coerce_paddleocr_option_indexes(response, self.options),
            [2],
        )


class OpenAIConfigManagerTest(unittest.TestCase):
    def test_paddleocr_env_counts_as_ai_config(self):
        with TemporaryDirectory() as workdir, patch.dict(
            "os.environ",
            {
                "PADDLEOCR_API_URL": "https://example.test/api/v2/ocr/jobs",
                "PADDLEOCR_API_TOKEN": "token",
            },
            clear=True,
        ):
            manager = OpenAIConfigManager(workdir)
            cfg = manager.load_config()

            self.assertTrue(manager.has_config())

        self.assertEqual(cfg, {"api_key": ""})


if __name__ == "__main__":
    unittest.main()


class CallbackFallbackTest(unittest.TestCase):
    def test_channel_invalid_is_treated_as_confirmation_fallback(self):
        self.assertTrue(
            _is_callback_confirmation_unavailable(
                RuntimeError("Telegram says: [400 CHANNEL_INVALID] - invalid channel")
            )
        )

    def test_unrelated_bad_request_is_not_treated_as_confirmation_fallback(self):
        self.assertFalse(
            _is_callback_confirmation_unavailable(
                RuntimeError("Telegram says: [400 MESSAGE_NOT_MODIFIED]")
            )
        )


class TerminalSuccessDetectionTest(unittest.TestCase):
    def setUp(self):
        self.signer = object.__new__(UserSigner)

    def test_detects_today_terminal_success_message(self):
        message = SimpleNamespace(
            text="🎉 签到成功，获得了 9积分",
            caption=None,
            date=datetime.now(timezone.utc),
        )

        self.assertTrue(self.signer._message_is_today_terminal_success(message))

    def test_ignores_old_terminal_success_message(self):
        message = SimpleNamespace(
            text="🎉 签到成功，获得了 9积分",
            caption=None,
            date=datetime.now(timezone.utc) - timedelta(days=1),
        )

        self.assertFalse(self.signer._message_is_today_terminal_success(message))

    def test_skips_verification_error_image(self):
        message = SimpleNamespace(text=None, caption="验证码错误!")

        self.assertTrue(self.signer._message_is_verification_error_image(message))

    def test_normalizes_verification_code_result(self):
        action = ReplyByImageRecognitionAction(
            ai_prompt="Read only the verification code from the image."
        )
        message = SimpleNamespace(text=None, caption="请输入验证码(不区分大小写):")

        self.assertEqual(
            self.signer._normalize_image_recognition_text(action, message, " b x t G "),
            "bxtG",
        )

    def test_rejects_brand_text_as_verification_code(self):
        action = ReplyByImageRecognitionAction(
            ai_prompt="Read only the verification code from the image."
        )
        message = SimpleNamespace(text=None, caption="请输入验证码(不区分大小写):")

        self.assertIsNone(
            self.signer._normalize_image_recognition_text(
                action, message, "EMBY PUBLIC\n\n# Peach"
            )
        )

    def test_extracts_code_candidate_from_ocr_context(self):
        action = ReplyByImageRecognitionAction(
            ai_prompt="Read only the verification code from the image."
        )
        message = SimpleNamespace(text=None, caption="请输入验证码(不区分大小写):")

        self.assertEqual(
            self.signer._normalize_image_recognition_text(
                action, message, "EMBY PUBLIC\n# Peach\nbxtG"
            ),
            "bxtG",
        )


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class AIToolsJsonFallbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_choose_options_by_image_requires_paddleocr(self):
        fake_completions = _FakeCompletions([])
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )
        tools = AITools({"api_key": "test", "model": "gpt-4o"})
        tools.client = fake_client

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PADDLEOCR_API_TOKEN"):
                await tools.choose_options_by_image(
                    b"fake-image",
                    "Choose the correct option",
                    [(1, "apple"), (2, "banana")],
                )

        self.assertEqual(fake_completions.calls, [])

    async def test_extract_text_by_image_requires_paddleocr(self):
        fake_completions = _FakeCompletions([])
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )
        tools = AITools({"api_key": "test", "model": "gpt-4o"})
        tools.client = fake_client

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PADDLEOCR_API_TOKEN"):
                await tools.extract_text_by_image(b"fake-image")

        self.assertEqual(fake_completions.calls, [])

    async def test_choose_options_by_image_uses_paddleocr_when_configured(self):
        fake_completions = _FakeCompletions([])
        tools = AITools({"api_key": "test", "model": "gpt-4o"})
        tools.client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        async def fake_request(_image, **_kwargs):
            return {
                "result": {
                    "layoutParsingResults": [
                        {"markdown": {"text": "banana"}},
                    ]
                }
            }

        tools._request_paddleocr = fake_request
        with patch.dict(
            "os.environ",
            {
                "PADDLEOCR_API_URL": "https://example.test/api/v2/ocr/jobs",
                "PADDLEOCR_API_TOKEN": "token",
            },
        ):
            result = await tools.choose_options_by_image(
                b"fake-image",
                "Choose the correct option",
                [(1, "apple"), (2, "banana")],
            )

        self.assertEqual(result, [2])
        self.assertEqual(fake_completions.calls, [])
