import logging
import os
from dotenv import load_dotenv
import streamlit as st
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

# Muat variabel lingkungan dari file .env
load_dotenv()

logger = logging.getLogger(__name__)

def get_ice_servers():
    """Gunakan server TURN Twilio karena Streamlit Community Cloud telah mengubah
    infrastrukturnya dan koneksi WebRTC tidak bisa terhubung tanpa server TURN sekarang."""

    try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        if not account_sid or not auth_token:
            raise KeyError("Twilio credentials are missing")
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    try:
        token = client.tokens.create()
    except TwilioRestException as e:
        st.warning(
            f"Error occurred while accessing Twilio API. Fallback to a free STUN server from Google. ({e})"
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    return token.ice_servers
