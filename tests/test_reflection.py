"""Task 8 integration checks with the real SDK and an offline HTTP transport.

Scripted revisions prove request/log/history wiring, not live model quality.
"""
import json
import unittest
from unittest.mock import patch

import test_agent_interaction as interaction
from test_agent_interaction import completion, tool_call


class ReflectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        interaction.AgentInteractionTests.setUpClass()
        cls.ns = interaction.AgentInteractionTests.ns

    def setUp(self):
        self.fixture = interaction.AgentInteractionTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.requests = self.fixture.requests
        self.history = []

    def client(self, *responses):
        return self.fixture.client(*responses)

    def chat(self, message, client, **options):
        return self.ns['chat'](message, self.history, api_client=client, **options)

    def records(self):
        return [json.loads(line) for line in self.fixture.logs.getvalue().splitlines()]

    def test_default_review_replaces_draft_with_same_rules_and_references(self):
        draft = 'Appointment automation costs $99 per month.'
        revised = 'Pricing requires a custom quote after discovery.'
        client = self.client(completion(draft), completion(revised))
        self.assertEqual(self.chat('How much does it cost?', client, model='configured-test-model'), revised)
        self.assertEqual(len(self.requests), 2)
        initial, review = self.requests
        self.assertEqual(review['model'], initial['model'])
        self.assertEqual(review['model'], 'configured-test-model')
        self.assertEqual(review['messages'][0], initial['messages'][0])
        self.assertIn(self.ns['business_summary'], review['messages'][0]['content'])
        self.assertIn(self.ns['business_pdf_text'], review['messages'][0]['content'])
        self.assertIn('missing email', review['messages'][1]['content'])
        self.assertEqual(review['messages'][-2], {'role': 'assistant', 'content': draft})
        self.assertEqual(review['tool_choice'], 'none')
        self.assertNotIn('tools', review)
        self.assertFalse(review['store'])
        self.assertEqual(self.history, [initial['messages'][-1], {'role': 'assistant', 'content': revised}])
        self.assertEqual(self.records(), [{
            'event': 'reply_reflection', 'status': 'success', 'draft': draft,
            'revised': revised, 'changed': True,
        }])
        self.assertFalse(client.is_closed())

    def test_missing_email_revision_and_unchanged_arabic_reply_are_logged(self):
        examples = [
            ("I'm Mira; I'd like appointment reminders.", 'We have everything we need!',
             'Thanks, Mira. What email address should we record?'),
            ('شو الخدمات اللي بتقدموها؟', 'منساعدك بأتمتة تذكير المواعيد.',
             'منساعدك بأتمتة تذكير المواعيد.'),
        ]
        for message, draft, revised in examples:
            with self.subTest(message=message):
                self.history.clear()
                client = self.client(completion(draft), completion(revised))
                self.assertEqual(self.chat(message, client), revised)
                self.assertEqual(self.history[-1]['content'], revised)
                record = self.records()[-1]
                self.assertEqual(record['changed'], draft != revised)
                self.assertEqual(record['revised'], revised)

    def test_review_sees_tools_and_prior_details_without_repeating_logs(self):
        lead = tool_call('record_customer_interest', {
            'email': 'mira@example.com', 'name': 'Mira', 'message': 'Appointment reminders'}, 'lead_review')
        draft = 'Your project is booked for tomorrow!'
        revised = 'Your interest was recorded, Mira.'
        client = self.client(completion(calls=[lead]), completion(draft), completion(revised),
                             completion('You are booked!'), completion('Your details were already recorded.'))
        message = "I'm Mira, mira@example.com; I'd like appointment reminders."
        self.assertEqual(self.chat(message, client), revised)
        review = self.requests[2]['messages']
        self.assertTrue(any(m['role'] == 'user' and m['content'] == message for m in review))
        result, = [m for m in review if m['role'] == 'tool']
        self.assertEqual(result['tool_call_id'], 'lead_review')
        self.assertEqual(json.loads(result['content'])['status'], 'success')
        self.assertEqual(self.chat('Do you have my email?', client), 'Your details were already recorded.')
        next_input = self.requests[3]['messages']
        self.assertIn({'role': 'assistant', 'content': revised}, next_input)
        self.assertNotIn(draft, str(next_input))
        self.assertNotIn(self.ns['REFLECTION_INSTRUCTIONS'], str(next_input))
        self.assertEqual(len([r for r in self.records() if r.get('event') == 'reply_reflection']), 2)
        self.assertEqual(len(self.records()), 3)  # One tool log plus two reviews.
        self.ns['_copy_chat_history'](self.history)

    def test_failed_tool_result_is_available_to_review(self):
        call = tool_call('record_customer_interest', {
            'email': 'invalid', 'name': 'Mira', 'message': 'Reminders'})
        client = self.client(completion(calls=[call]), completion('Your interest was recorded.'),
                             completion('I could not record it. What valid email should we use?'))
        self.assertIn('could not record', self.chat('Please record my interest.', client))
        result, = [m for m in self.requests[-1]['messages'] if m['role'] == 'tool']
        self.assertEqual(json.loads(result['content'])['status'], 'error')
        self.assertEqual(len(self.records()), 1)  # No successful lead log.

    def test_review_errors_withhold_draft_and_preserve_completed_tools(self):
        failures = [429, completion(), completion('   '),
                    {**completion(), 'choices': []},
                    completion('Incomplete', finish_reason='length'),
                    completion('Filtered', finish_reason='content_filter'),
                    completion(calls=[tool_call(call_id='must_not_execute')])]
        for failure in failures:
            with self.subTest(failure=failure):
                self.history.clear()
                self.fixture.logs.seek(0)
                self.fixture.logs.truncate()
                client = self.client(completion(calls=[tool_call()]), completion('Unreviewed draft'), failure)
                with self.assertRaisesRegex(RuntimeError, 'Reply reflection failed'):
                    self.chat('Business hours?', client)
                self.assertEqual(self.history[-1]['role'], 'tool')
                self.assertNotIn('Unreviewed draft', str(self.history))
                self.ns['_copy_chat_history'](self.history)
                records = self.records()
                self.assertEqual(len(records), 2)  # One tool execution, one failed review.
                self.assertEqual(records[-1]['status'], 'error')
                self.assertEqual(records[-1]['draft'], 'Unreviewed draft')
                self.assertIsNone(records[-1]['revised'])

    def test_disabling_review_and_invalid_option(self):
        client = self.client(completion('Unreviewed comparison reply'))
        self.assertEqual(self.chat('Hello', client, reflect=False), 'Unreviewed comparison reply')
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.records(), [])
        self.history.clear()
        for invalid in (None, 1, 0, 'yes'):
            with self.subTest(reflect=invalid):
                with self.assertRaisesRegex(ValueError, 'reflect must be'):
                    self.chat('Hello', client, reflect=invalid)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.history, [])

    def test_owned_client_is_closed_even_when_review_fails(self):
        client = self.client(completion('Draft'), 401)
        with patch.dict(self.ns, {'create_openai_client': lambda: client}):
            with self.assertRaisesRegex(RuntimeError, 'Reply reflection failed'):
                self.ns['chat']('Hello', self.history)
        self.assertTrue(client.is_closed())

    def test_gradio_default_chat_returns_only_reviewed_reply(self):
        client = self.client(completion('Draft with invented price'), completion('A custom quote is required.'))
        with patch.dict(self.ns, {'create_openai_client': lambda: client}):
            reply, state = self.ns['gradio_reply']('How much?', [], None)
        self.assertEqual(reply, 'A custom quote is required.')
        self.assertEqual(state['messages'][-1]['content'], reply)
        self.assertNotIn('Draft with invented price', str(state))
        self.assertEqual(state['turn_ends'], [2])
        self.assertEqual(len(self.requests), 2)


if __name__ == '__main__':
    unittest.main()
