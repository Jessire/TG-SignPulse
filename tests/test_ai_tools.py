import unittest
from datetime import datetime, timedelta, timezone
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from tg_signer.ai_tools import AITools
from tg_signer.config import (
    ChooseOptionByImageAction,
    ClickKeyboardByTextAction,
    ReplyByImageRecognitionAction,
    SignChatV3,
)
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

    def test_timeout_is_treated_as_transient_ai_error(self):
        self.assertTrue(AITools._should_retry_transient_ai_error(TimeoutError()))

    def test_quota_exhaustion_is_not_retried_as_transient_error(self):
        error = RuntimeError(
            "Error code: 429 - {'error': {'status': 'RESOURCE_EXHAUSTED', "
            "'message': 'You exceeded your current quota, free_tier'}}"
        )

        self.assertFalse(AITools._should_retry_transient_ai_error(error))

    def test_prepare_text_extraction_images_adds_red_captcha_variant(self):
        image = Image.new("RGB", (320, 120), (93, 75, 210))
        draw = ImageDraw.Draw(image)
        draw.text((40, 28), "IkKR", fill=(150, 30, 38))

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        variants = AITools._prepare_text_extraction_images(buffer.getvalue())

        self.assertGreaterEqual(len(variants), 2)
        with Image.open(BytesIO(variants[0])) as prepared:
            self.assertEqual(prepared.mode, "RGB")
            gray = prepared.convert("L")
            histogram = gray.histogram()
            dark_pixels = sum(histogram[:64])
            light_pixels = sum(histogram[221:])
            self.assertGreater(dark_pixels, 20)
            self.assertGreater(light_pixels, dark_pixels * 5)


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

    def test_strong_success_overrides_prior_verification_error_text(self):
        message = SimpleNamespace(
            text=None,
            caption="验证码错误!\n🎉 签到成功，获得了 20积分\n💰总积分：1563",
            date=datetime.now(timezone.utc),
        )

        self.assertTrue(self.signer._message_is_today_terminal_success(message))


class _FakeHistoryApp:
    def __init__(self, messages):
        self.messages = messages

    def get_chat_history(self, chat_id, limit):
        async def generate():
            for message in self.messages[:limit]:
                yield message

        return generate()


class WaitForTerminalSuccessTest(unittest.IsolatedAsyncioTestCase):
    async def test_click_button_wait_stops_on_explicit_today_success_without_date(self):
        signer = object.__new__(UserSigner)
        signer.context = signer.ensure_ctx()
        signer.log = lambda *args, **kwargs: None
        chat = SignChatV3(
            chat_id=8468306814,
            name="EmbyPulse",
            actions=[ClickKeyboardByTextAction(text="每日签到")],
        )
        message = SimpleNamespace(
            id=300,
            chat=SimpleNamespace(id=chat.chat_id),
            text="😊 今天已经签到过了，明天再来吧！",
            caption=None,
            photo=None,
            media=None,
            reply_markup=None,
            date=None,
            edit_date=None,
            message_thread_id=None,
            reply_to_top_message_id=None,
        )
        signer.app = _FakeHistoryApp([message])

        result = await signer.wait_for(
            chat,
            ClickKeyboardByTextAction(text="每日签到"),
            timeout=0.05,
            before_action_state={},
        )

        self.assertTrue(result)
        self.assertTrue(signer.context.stop_after_current_action)

    async def test_wait_for_accepts_existing_today_terminal_success_from_history(self):
        signer = object.__new__(UserSigner)
        signer.context = signer.ensure_ctx()
        signer.log = lambda *args, **kwargs: None
        chat = SignChatV3(
            chat_id=8060839337,
            name="Peach Emby",
            actions=[ReplyByImageRecognitionAction()],
        )
        message = SimpleNamespace(
            id=99,
            chat=SimpleNamespace(id=chat.chat_id),
            text=None,
            caption="🎉 签到成功，获得了 20积分",
            photo=SimpleNamespace(file_id="photo"),
            media=None,
            reply_markup=None,
            date=datetime.now(timezone.utc),
            edit_date=None,
            message_thread_id=None,
            reply_to_top_message_id=None,
        )
        signer.app = _FakeHistoryApp([message])
        before_action_state = {message.id: signer._message_state_marker(message)}

        result = await signer.wait_for(
            chat,
            ReplyByImageRecognitionAction(),
            timeout=0.05,
            before_action_state=before_action_state,
        )

        self.assertIsNone(result)
        self.assertTrue(signer.context.stop_after_current_action)

    async def test_image_reply_waits_for_terminal_success_after_answer(self):
        signer = object.__new__(UserSigner)
        signer.context = signer.ensure_ctx()
        signer.log = lambda *args, **kwargs: None
        chat = SignChatV3(
            chat_id=8060839337,
            name="Peach Emby",
            actions=[ReplyByImageRecognitionAction()],
        )
        captcha = SimpleNamespace(
            id=100,
            chat=SimpleNamespace(id=chat.chat_id),
            text=None,
            caption="请输入验证码(不区分大小写):",
            photo=SimpleNamespace(file_id="captcha"),
            media=None,
            reply_markup=None,
            date=datetime.now(timezone.utc),
            edit_date=None,
            message_thread_id=None,
            reply_to_top_message_id=None,
        )
        success = SimpleNamespace(
            id=101,
            chat=SimpleNamespace(id=chat.chat_id),
            text=None,
            caption="🎉 签到成功，获得了 16积分\n💰 总积分：1593",
            photo=SimpleNamespace(file_id="success"),
            media=None,
            reply_markup=None,
            date=datetime.now(timezone.utc),
            edit_date=None,
            message_thread_id=None,
            reply_to_top_message_id=None,
        )
        signer.context.chat_messages[chat.chat_id][captcha.id] = captcha
        signer.app = _FakeHistoryApp([success, captcha])

        async def reply_ok(action, message):
            return True

        signer._reply_by_image_recognition = reply_ok

        result = await signer.wait_for(
            chat,
            ReplyByImageRecognitionAction(),
            timeout=1.0,
            before_action_state={},
        )

        self.assertIsNone(result)
        self.assertTrue(signer.context.stop_after_current_action)

    async def test_image_button_click_waits_for_terminal_success_after_answer(self):
        signer = object.__new__(UserSigner)
        signer.context = signer.ensure_ctx()
        signer.log = lambda *args, **kwargs: None
        chat = SignChatV3(
            chat_id=1429576125,
            name="Emby Public",
            actions=[ChooseOptionByImageAction()],
        )
        challenge = SimpleNamespace(
            id=200,
            chat=SimpleNamespace(id=chat.chat_id),
            text=None,
            caption="请在 30 秒内点击图中事物的按钮以完成签到",
            photo=SimpleNamespace(file_id="challenge"),
            media=None,
            reply_markup=None,
            date=datetime.now(timezone.utc),
            edit_date=None,
            message_thread_id=None,
            reply_to_top_message_id=None,
        )
        success = SimpleNamespace(
            id=201,
            chat=SimpleNamespace(id=chat.chat_id),
            text="🎉 签到成功！\n🎲 获得 53 积分",
            caption=None,
            photo=None,
            media=None,
            reply_markup=None,
            date=datetime.now(timezone.utc),
            edit_date=None,
            message_thread_id=None,
            reply_to_top_message_id=None,
        )
        signer.context.chat_messages[chat.chat_id][challenge.id] = challenge
        signer.app = _FakeHistoryApp([success, challenge])

        async def click_ok(action, message):
            return True

        signer._choose_option_by_image = click_ok

        result = await signer.wait_for(
            chat,
            ChooseOptionByImageAction(),
            timeout=1.0,
            before_action_state={},
        )

        self.assertIsNone(result)
        self.assertTrue(signer.context.stop_after_current_action)


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
    @staticmethod
    def _red_captcha_bytes():
        image = Image.new("RGB", (320, 120), (93, 75, 210))
        draw = ImageDraw.Draw(image)
        draw.text((40, 28), "IkKR", fill=(150, 30, 38))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    async def test_extract_text_by_image_sends_only_captcha_variant_first(self):
        fake_completions = _FakeCompletions(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="IkKR"))
                    ]
                ),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )
        tools = AITools({"api_key": "test", "model": "gpt-4o"})
        tools.client = fake_client

        result = await tools.extract_text_by_image(self._red_captcha_bytes())

        self.assertEqual(result, "IkKR")
        content = fake_completions.calls[0]["messages"][1]["content"]
        image_parts = [part for part in content if part["type"] == "image_url"]
        self.assertEqual(len(image_parts), 1)

    async def test_extract_text_by_image_tries_variants_one_at_a_time_when_empty(self):
        fake_completions = _FakeCompletions(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content=""))
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="IkKR"))
                    ]
                ),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )
        tools = AITools({"api_key": "test", "model": "gpt-4o"})
        tools.client = fake_client

        with patch.dict(
            "os.environ",
            {"AI_VISION_RETRY_ATTEMPTS": "2", "AI_VISION_RETRY_DELAY": "0"},
        ):
            result = await tools.extract_text_by_image(self._red_captcha_bytes())

        self.assertEqual(result, "IkKR")
        self.assertEqual(len(fake_completions.calls), 2)
        for call in fake_completions.calls:
            content = call["messages"][1]["content"]
            image_parts = [part for part in content if part["type"] == "image_url"]
            self.assertEqual(len(image_parts), 1)

    async def test_zhipu_base_url_sends_raw_base64_image_url(self):
        for base_url in (
            "https://open.bigmodel.cn/api/paas/v4",
            "https://api.z.ai/api/paas/v4",
        ):
            with self.subTest(base_url=base_url):
                fake_completions = _FakeCompletions(
                    [
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content='{"options":[1]}')
                                )
                            ]
                        ),
                    ]
                )
                fake_client = SimpleNamespace(
                    chat=SimpleNamespace(completions=fake_completions)
                )
                tools = AITools(
                    {
                        "api_key": "test",
                        "base_url": base_url,
                        "model": "GLM-4.6V-Flash",
                    }
                )
                tools.client = fake_client

                await tools.choose_options_by_image(
                    b"fake-image",
                    "Choose the correct option",
                    [(1, "apple"), (2, "banana")],
                )

                image_url = fake_completions.calls[0]["messages"][1]["content"][1]["image_url"]["url"]
                self.assertEqual(image_url, "ZmFrZS1pbWFnZQ==")

    async def test_standard_base_url_sends_data_url_image_url(self):
        fake_completions = _FakeCompletions(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"options":[1]}')
                        )
                    ]
                ),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )
        tools = AITools(
            {
                "api_key": "test",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o",
            }
        )
        tools.client = fake_client

        await tools.choose_options_by_image(
            b"fake-image",
            "Choose the correct option",
            [(1, "apple"), (2, "banana")],
        )

        image_url = fake_completions.calls[0]["messages"][1]["content"][1]["image_url"]["url"]
        self.assertEqual(image_url, "data:image/jpeg;base64,ZmFrZS1pbWFnZQ==")

    async def test_choose_options_by_image_retries_without_json_mode(self):
        fake_completions = _FakeCompletions(
            [
                RuntimeError("Error code: 403 - {'message': 'openai_error', 'code': 'bad_response_status_code', 'detail': 'response_format json_object unsupported'}"),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"options":[2]}')
                        )
                    ]
                ),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )
        tools = AITools({"api_key": "test", "model": "gpt-4o"})
        tools.client = fake_client

        result = await tools.choose_options_by_image(
            b"fake-image",
            "Choose the correct option",
            [(1, "apple"), (2, "banana")],
        )

        self.assertEqual(result, [2])
        self.assertIn("response_format", fake_completions.calls[0])
        self.assertNotIn("response_format", fake_completions.calls[1])

    async def test_choose_options_by_image_retries_transient_provider_errors(self):
        fake_completions = _FakeCompletions(
            [
                RuntimeError("Error code: 503 - {'error': {'status': 'UNAVAILABLE'}}"),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"options":[2]}')
                        )
                    ]
                ),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )
        tools = AITools({"api_key": "test", "model": "gpt-4o"})
        tools.client = fake_client

        result = await tools.choose_options_by_image(
            b"fake-image",
            "Choose the correct option",
            [(1, "apple"), (2, "banana")],
        )

        self.assertEqual(result, [2])
        self.assertEqual(len(fake_completions.calls), 2)

    async def test_extract_text_by_image_retries_transient_provider_errors(self):
        fake_completions = _FakeCompletions(
            [
                RuntimeError("Error code: 503 - {'error': {'status': 'UNAVAILABLE'}}"),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="bxtG"))
                    ]
                ),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )
        tools = AITools({"api_key": "test", "model": "gpt-4o"})
        tools.client = fake_client

        result = await tools.extract_text_by_image(b"fake-image")

        self.assertEqual(result, "bxtG")
        self.assertEqual(len(fake_completions.calls), 2)

    async def test_extract_text_by_image_retries_empty_provider_response(self):
        fake_completions = _FakeCompletions(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content=""))
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="IkKR"))
                    ]
                ),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )
        tools = AITools({"api_key": "test", "model": "gpt-4o"})
        tools.client = fake_client

        result = await tools.extract_text_by_image(b"fake-image")

        self.assertEqual(result, "IkKR")
        self.assertEqual(len(fake_completions.calls), 2)
