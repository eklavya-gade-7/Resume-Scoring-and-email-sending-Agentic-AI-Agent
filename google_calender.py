import os

from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/calendar"]

TIMEZONE = ZoneInfo("Asia/Kolkata")


SLOT_MINUTES = 45
SLOT_HOURS = [10, 14, 16, 18]
SLOT_COUNT = 6
SLOT_LEAD_MINUTES = 60


def build_proposed_slots(count=SLOT_COUNT, hours=SLOT_HOURS, minutes=SLOT_MINUTES,
                         lead_minutes=SLOT_LEAD_MINUTES):
    """The next few interview slots, counted forward from right now.

    These used to be written out by hand with fixed dates, which went stale -
    by the time the pipeline ran, the first slot of the day had already passed
    and candidates were still being offered it. Building them from now means
    every slot is always in the future, whenever the run happens.

    The first slot is the next half hour that is at least lead_minutes away, so
    a candidate is never offered a time they could not realistically make. The
    rest are the standing hours on today and the days after it.

    Today counts, and so do weekends. Skipping them pushed the first offer up
    to two days out when the pipeline ran on a Saturday, which made the whole
    scheduling step look inert.
    """
    now = datetime.now(TIMEZONE)

    earliest = (now + timedelta(minutes=lead_minutes)).replace(second=0, microsecond=0)

    # Round up to the next half hour so the soonest slot reads as a clean time.
    if earliest.minute % 30:
        earliest = earliest + timedelta(minutes=30 - earliest.minute % 30)

    starts = []

    # Only offer the soonest slot if it lands inside the working window. Run
    # this at 23:40 and the next half hour is 01:00, which is not an interview.
    if time(hours[0], 0) <= earliest.timetz().replace(tzinfo=None) <= time(hours[-1], 0):
        starts.append(earliest)

    day = now.date()

    while len(starts) < count:

        for hour in hours:

            start = datetime.combine(day, time(hour, 0), tzinfo=TIMEZONE)

            if start <= earliest:
                continue

            # Two slots less than one slot-length apart would overlap, and a
            # candidate should never be shown a choice between 13:30 and 14:00.
            if any(abs((start - chosen).total_seconds()) < minutes * 60
                   for chosen in starts):
                continue

            starts.append(start)

        day = day + timedelta(days=1)

    slots = []

    for start in sorted(starts)[:count]:

        end = start + timedelta(minutes=minutes)

        slots.append({
            "start": start.isoformat(),
            "end": end.isoformat()
        })

    return slots


proposed_slots = build_proposed_slots()


def print_upcoming_events(service):

    now = datetime.now(timezone.utc).isoformat()

    events_result = service.events().list(calendarId="primary", timeMin=now, maxResults=10, singleEvents=True, orderBy="startTime").execute()

    events = events_result.get("items", [])

    if len(events) == 0:
        print("No upcoming events found.")
        return

    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        print(start, event.get("summary", "No title"))


def get_calendar_service():

    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)

    return service


def get_busy_periods(service, start_datetime, end_datetime):

    body = {
        "timeMin": start_datetime.isoformat(),
        "timeMax": end_datetime.isoformat(),
        "timeZone": "Asia/Kolkata",
        "items": [{"id": "primary"}]
    }

    response = service.freebusy().query(body=body).execute()

    busy_periods = response["calendars"]["primary"]["busy"]

    return busy_periods


def get_free_slots(busy_periods, proposed_slots):

    free_slots = []

    now = datetime.now(TIMEZONE)

    for free_slot in proposed_slots:

        flag = 0

        free_slot_start_datetime = datetime.fromisoformat(free_slot["start"])
        free_slot_end_datetime = datetime.fromisoformat(free_slot["end"])

        # A slot that has already started is not free, however empty the
        # calendar looks. Without this a stale slot list offers times in the past.
        if free_slot_start_datetime <= now:
            continue

        for busy_slot in busy_periods:

            busy_slot_start_datetime = datetime.fromisoformat(busy_slot["start"])
            busy_slot_end_datetime = datetime.fromisoformat(busy_slot["end"])

            if free_slot_start_datetime < busy_slot_end_datetime and free_slot_end_datetime > busy_slot_start_datetime:
                flag = 1
                break

        if flag == 0:
            free_slots.append(free_slot)

    return free_slots


def insert_slot(service, selected_slot):

    event_body = {
        "start": {
            "dateTime": selected_slot["start"],
            "timeZone": "Asia/Kolkata"
        },
        "end": {
            "dateTime": selected_slot["end"],
            "timeZone": "Asia/Kolkata"
        }
    }

    event = service.events().insert(calendarId="primary", body=event_body).execute()

    print("Event created:", event.get("htmlLink"))

    return event


if __name__ == "__main__":

    service = get_calendar_service()

    start_datetime = min(datetime.fromisoformat(slot["start"]) for slot in proposed_slots)
    end_datetime = max(datetime.fromisoformat(slot["end"]) for slot in proposed_slots)

    busy_periods = get_busy_periods(service, start_datetime, end_datetime)
    free_slots = get_free_slots(busy_periods, proposed_slots)

    print("Busy periods:")
    print(busy_periods)

    print("Free slots:")
    print(free_slots)