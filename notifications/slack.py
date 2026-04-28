import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SlackNotifier:
    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")

    def send_alert(self, fault_name, run_name, mse, confidence, explanation=None):
        """
        Sends a formatted alert to Slack
        """
        if not self.webhook_url:
            print("⚠️ [Slack] No webhook URL found in .env. Skipping alert.")
            return

        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🚨 Anomaly Detected: " + fault_name,
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Run ID:*\n{run_name}"},
                        {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.2f}"},
                        {"type": "mrkdwn", "text": f"*MSE Score:*\n{mse:.4f}"},
                        {"type": "mrkdwn", "text": f"*Status:*\n{fault_name}"}
                    ]
                }
            ]
        }

        if explanation:
            payload["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*AI Analysis:*\n{explanation}"
                }
            })

        try:
            response = requests.post(
                self.webhook_url, 
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code != 200:
                print(f"❌ [Slack] Failed to send alert: {response.status_code}, {response.text}")
            else:
                print(f"✅ [Slack] Alert sent for {run_name}")
        except Exception as e:
            print(f"❌ [Slack] Error sending alert: {e}")

if __name__ == "__main__":
    # Test (requires SLACK_WEBHOOK_URL in .env)
    notifier = SlackNotifier()
    notifier.send_alert("UNKNOWN FAULT", "RUN_TEST_001", 0.85, 0.45, "This is a test explanation.")
