"""Offline checks of the Gradio adapter and its session history."""
from copy import deepcopy
import os
import unittest
from unittest.mock import patch

import test_agent_interaction as interaction
from test_agent_interaction import completion, tool_call


class GradioInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        interaction.AgentInteractionTests.setUpClass()
        cls.ns = interaction.AgentInteractionTests.ns

    def reply(self, message, history, state, agent):
        return self.ns['gradio_reply'](message, history, state, agent_fn=agent)

    @staticmethod
    def agent(message, history):
        answer = f'Reply {sum(m["role"] == "user" for m in history) + 1}: {message}'
        history.extend([{'role': 'user', 'content': message},
                        {'role': 'assistant', 'content': answer}])
        return answer

    @staticmethod
    def visible(*messages):
        # Gradio 6 uses normalized content blocks in its displayed history.
        return [entry for message in messages for entry in (
            {'role': 'user', 'content': [{'type': 'text', 'text': message}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Reply'}]},
        )]

    def test_sessions_are_independent_and_inputs_are_not_mutated(self):
        initial = self.ns['new_ui_session']()
        _, first = self.reply('Mira', [], initial, self.agent)
        self.assertEqual(initial, {'messages': [], 'turn_ends': []})
        snapshot = deepcopy(first)
        reply, second = self.reply('Remember me?', self.visible('Mira'), first, self.agent)
        self.assertEqual(reply, 'Reply 2: Remember me?')
        self.assertEqual(first, snapshot)
        other_reply, other = self.reply('Sam', [], None, self.agent)
        self.assertEqual(other_reply, 'Reply 1: Sam')
        self.assertNotIn('Mira', str(other))
        self.assertEqual(len(second['turn_ends']), 2)

    def test_retry_and_undo_remove_discarded_hidden_turns(self):
        _, first = self.reply('first', [], None, self.agent)
        _, second = self.reply('discarded', self.visible('first'), first, self.agent)
        reply, retried = self.reply('replacement', self.visible('first'), second, self.agent)
        self.assertEqual(reply, 'Reply 2: replacement')
        self.assertNotIn('discarded', str(retried))
        reply, restarted = self.reply('new start', [], retried, self.agent)
        self.assertEqual(reply, 'Reply 1: new start')
        self.assertNotIn('replacement', str(restarted))

    def test_missing_key_returns_readable_error_and_valid_history(self):
        with patch.dict(os.environ, {}, clear=True):
            reply, state = self.ns['gradio_reply']('Hello', [], None)
        self.assertIn('OPENAI_API_KEY', reply)
        self.assertEqual([m['role'] for m in state['messages']], ['user', 'assistant'])
        self.ns['_copy_chat_history'](state['messages'])
        reply, state = self.reply('Try again', self.visible('Hello'), state, self.agent)
        self.assertEqual(reply, 'Reply 2: Try again')

    def test_completed_tools_survive_errors_and_follow_up(self):
        # Use the real SDK/agent loop with an in-process HTTP transport.
        fixture = interaction.AgentInteractionTests(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        client = fixture.client(completion(calls=[tool_call()]), 429,
                                completion('Your question was already recorded.'))
        def agent(message, history):
            return self.ns['chat'](message, history, api_client=client, reflect=False)
        reply, state = self.reply('Business hours?', [], None, agent)
        self.assertIn('RateLimitError', reply)
        self.assertTrue(any(m['role'] == 'tool' for m in state['messages']))
        self.ns['_copy_chat_history'](state['messages'])
        reply, state = self.reply('Was it recorded?', self.visible('Business hours?'), state, agent)
        self.assertEqual(reply, 'Your question was already recorded.')
        self.assertTrue(any(m['role'] == 'tool' for m in fixture.requests[-1]['messages']))
        self.assertEqual(len(fixture.logs.getvalue().splitlines()), 1)

    def test_clear_binding_resets_agent_state(self):
        demo = self.ns['build_gradio_interface'](self.agent)
        self.addCleanup(demo.close)
        callbacks = [d for d in demo.config['dependencies']
                     if d['api_name'] == 'new_ui_session'
                     and any(event == 'clear' for _, event in d['targets'])]
        self.assertEqual(len(callbacks), 1)
        state_id, = callbacks[0]['outputs']
        self.assertTrue(any(c['id'] == state_id and c['type'] == 'state'
                            for c in demo.config['components']))
        reset = demo.fns[callbacks[0]['id']].fn
        self.assertEqual(reset(), {'messages': [], 'turn_ends': []})
        reply, state = self.reply('Fresh chat', [], reset(), self.agent)
        self.assertEqual(reply, 'Reply 1: Fresh chat')


if __name__ == '__main__':
    unittest.main()
