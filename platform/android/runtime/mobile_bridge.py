"""JSON entrypoint for the Android MethodChannel; no network or server."""
import json
from modelforge.mobile.offline import generate, train


def request(method, home, payload_json, available_mb):
    payload = json.loads(payload_json)
    if method == 'train':
        result = train(home, payload['purpose'], payload['text'], payload['target_weight_mb'],
                       payload['loop_passes'], available_mb)
    elif method == 'generate':
        result = {'text': generate(home, payload['prompt'], payload.get('max_chars', 64))}
    else:
        raise ValueError('Unknown offline action')
    return json.dumps(result, ensure_ascii=False)
