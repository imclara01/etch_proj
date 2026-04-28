from notifications.slack import SlackNotifier
import os
from dotenv import load_dotenv

load_dotenv()

def test_slack():
    print(f"Testing Slack notification...")
    print(f"Webhook URL: {os.getenv('SLACK_WEBHOOK_URL')}")
    notifier = SlackNotifier()
    notifier.send_alert("TEST FAULT", "RUN_TEST_MANUAL", 0.999, 0.95, "This is a manual test message to verify the connection.")

if __name__ == "__main__":
    test_slack()
