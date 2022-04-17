import os

from twilio.rest import Client

# To set up environmental variables, see http://twil.io/secure
ACCOUNT_SID = "ACc67a5fbe3477b94dadea58c9b19f0d24"
AUTH_TOKEN = "d67bc66969beafba9752247bd77f3161"

client = Client(ACCOUNT_SID, AUTH_TOKEN)
notification = client.messages.create(to = "+15715282308", from_ = "+19893498993",
#notification = client.notify.services('ISXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX') \
 #   .notifications.create(
  #      to_binding='{"binding_type":"sms", "address":"+19893498993"}',
        body='Knok-Knok! This is your first Notify SMS')
print(notification.sid)