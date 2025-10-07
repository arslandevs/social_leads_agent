
from flask import Flask, request, jsonify
import requests
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from socials_leads_agent.social_agent_v10 import graph, normalize_query  # Assuming the graph is defined in agent.py

app = Flask(__name__)


codestreaks_user_token = "IGAAb1e5fjArxBZAE9VR25aQnhDOHZAvcmVpODBaSGQ5VUd2WW1DUmFIN1oySy1oak40UGxOZADVEbG5ibnRnR1cteGdraDFmWW9PUURuaEJ4WXJxZAEhFN29BWFVtMzRHX054Vm9JY09ldzNlLWFQbVZA3clhTbFJ6Q0NYZA2ZACaGJBNAZDZD"
IG_PAGE_ACCESS_TOKEN = "EAAUIuz7UXk4BPE99e0rKZCMVLgUGmCX3EZCNaUikyaAhWbob6hw5Ion8BcN3ZAgM8ZAcuGOzOLSXq6cCakeyOnaJJZADJfcHh7FTKCZAy1SDaHf5f6ZAWEWcv10ZB2mxQrlpz9HsKJHMvJF9fxyV6o22sg1GvYfBNmowZCS7RTFx4Cp6qBZAlbEZCZAwDef2WgYqAcJjcsnTCQJYgGgaTPFoN43ajzVS2pxLNQku"
VERIFY_TOKEN = "12345678"
PAGE_ID = "705096629359008"
IG_SCOPED_USER_ID = 17841404219800880

seen_mids = set()

# ============================================
# API BLOCK
# ============================================

def get_instagram_username(user_id):
    url = f"https://graph.facebook.com/v23.0/{user_id}"
    params = {
        "fields": "username",  # You can also request 'name', 'profile_pic' if available
        "access_token": IG_PAGE_ACCESS_TOKEN
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        return data.get("username")  # Or 'name' if you request that
    else:
        print("Error fetching username:", resp.status_code, resp.text)
        return None


@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        try:
            if mode == "subscribe" and token == VERIFY_TOKEN:
                return challenge, 200
            else:
                print(f"Verification failed: mode={mode}, token={token}")
                return "Forbidden", 403
        except Exception as e:
            print(f"Error during verification: {e}")
            return "Internal Server Error", 500

    # 2) Handle incoming messages
    data = request.json or {}
    events = data.get("entry", [{}])[0].get("messaging", [])

    for ev in events:
        msg = ev.get("message", {})
        mid = msg.get("mid")

        # Skip unwanted events
        if not mid or msg.get("is_echo") or msg.get("is_self") or mid in seen_mids:
            continue

        seen_mids.add(mid)
        text = msg.get("text", "").strip()
        if not text:
            continue

        sender_id = ev["sender"]["id"]

        # **Print what the user sent**
        print(f"=======================User ({sender_id}) sent: {text}")

        # Prepare reply
        # ai_reply_text = "hello"
        # session_id = str(uuid.uuid4())
        # Reuse the same session for this user so history persists
        
        username = get_instagram_username(sender_id)  # Fetch username from IG API
        print(f"\n\nUser ({username}) [{sender_id}] sent: {text}")

        # Pass username to agent as part of the conversation
        session_id = f"ig_user_{sender_id}"
        config = RunnableConfig(configurable={
            "session_id": session_id,
            "username": username  # Custom var for use in the agent
        })

        ai_reply_text = graph.invoke(
            {"messages": [HumanMessage(content=text)]}, config=config
        )["messages"]


        payload = {
            "messaging_type": "RESPONSE",
            "recipient": {"id": sender_id},
            "message": {"text": ai_reply_text[-1].content}  # Get the last message content
        }

        # payload = {
        #     "messaging_product": "instagram",
        #     "recipient": {"id": sender_id},   # from your IG webhook, not a FB PSID
        #     "message": {"text": "Thanks for reaching out!"}
        # }

        reply_url = f"https://graph.facebook.com/v23.0/me/messages?access_token={IG_PAGE_ACCESS_TOKEN}"

        # Print the outgoing request
        print("\n\n--- Outgoing request ---")
        print("POST", reply_url)
        print("Payload:", payload)

        # Send reply
        resp = requests.post(reply_url, json=payload)

        # Print the response
        print("\n\n--- Response ---")
        print("Status:", resp.status_code)
        print("Text:", resp.text)
        print("-----------------------\n")

    return jsonify(status="ok"), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
