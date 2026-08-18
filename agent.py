from typing import List, TypedDict
from datetime import datetime
from langgraph.types import interrupt, Command
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from google_calender import get_busy_periods, get_calendar_service, get_free_slots
from langgraph.graph import START, END, StateGraph
from email.message import EmailMessage
import smtplib
import os
from dotenv import load_dotenv

load_dotenv()

email_address = os.getenv("EMAIL_ADDRESS")
password = os.getenv("PASSWORD")


class candidate(TypedDict):
    name: str
    email: str
    score: float


class RecruitmentState(TypedDict):
    candidate: candidate
    free_slots: list
    proposed_slots: list
    selected_slot: dict
    email_draft: dict
    thread_id: str
    slot_available : bool


def prepare_candidate(state):

    candidate_data = state["candidate"]

    print(candidate_data["name"], candidate_data["email"], candidate_data["score"])

    return {"candidate": candidate_data}


def find_available_slots(state):

    proposed_slots = state["proposed_slots"]

    service = get_calendar_service()

    start_datetime = min(datetime.fromisoformat(slot["start"]) for slot in proposed_slots)
    end_datetime = max(datetime.fromisoformat(slot["end"]) for slot in proposed_slots)

    busy_periods = get_busy_periods(service, start_datetime, end_datetime)

    free_slots = get_free_slots(busy_periods, proposed_slots)

    if len(free_slots) == 0:
        print("  no free slots left, no invite sent to", state["candidate"]["name"])

    return {"free_slots": free_slots}


def add_slot_to_calendar(state):

    selected_slot = state["selected_slot"]
    candidate_data = state["candidate"]

    service = get_calendar_service()

    event = {
        "summary": f"Interview - {candidate_data['name']}",
        "start": {
            "dateTime": selected_slot["start"],
            "timeZone": "Asia/Kolkata"
        },
        "end": {
            "dateTime": selected_slot["end"],
            "timeZone": "Asia/Kolkata"
        }
    }

    service.events().insert(calendarId="primary", body=event).execute()

    return {}


def draft_interview_email(state):

    candidate_data = state["candidate"]
    free_slots = state["free_slots"]
    thread_id = state["thread_id"]

    slot_text = ""

    for i, slot in enumerate(free_slots):

        start = datetime.fromisoformat(slot["start"])
        end = datetime.fromisoformat(slot["end"])

        slot_text += f"{i + 1}. {start.strftime('%d %b %Y, %I:%M %p')} - {end.strftime('%I:%M %p')}\n"

    link = f"http://localhost:8501/?thread_id={thread_id}"

    subject = "Interview Invitation"

    body = f"""Hi {candidate_data["name"]},

You have been shortlisted for an interview.
Your resume score was {candidate_data["score"]}.

The following interview slots are currently available:

{slot_text}

Please select one of the available slots using the link below:

{link}

Regards,
Recruitment Team
"""

    email_draft = {
        "name": candidate_data["name"],
        "email": candidate_data["email"],
        "subject": subject,
        "body": body
    }

    return {"email_draft": email_draft}


def send_out_email(state):

    draft = state["email_draft"]

    message = EmailMessage()

    message["From"] = email_address
    message["To"] = draft["email"]
    message["Subject"] = draft["subject"]
    message.set_content(draft["body"])

    s = smtplib.SMTP("smtp.gmail.com", 587)
    s.starttls()
    s.login(email_address, password)
    s.send_message(message)
    s.quit()

    return {}


def select_slot(state):

    free_slots = state["free_slots"]

    chosen_slot = interrupt({
        "message": "Kindly choose your preferred interview slot",
        "free_slots": free_slots
    })

    selected_slot = free_slots[chosen_slot - 1]

    return {"selected_slot": selected_slot}

def recheck_selected_slot(state):
    slot = state["selected_slot"]
    
    slot_start = datetime.fromisoformat(slot["start"])
    slot_end = datetime.fromisoformat(slot["end"])
    
    service = get_calendar_service()
    busy_slots = get_busy_periods(service,slot_start,slot_end)
    
    if(len(busy_slots) == 0):
        return {"slot_available" : True}
    

    proposed_slots = state["proposed_slots"]

    start_datetime = min(datetime.fromisoformat(slot["start"]) for slot in proposed_slots)
    end_datetime = max(datetime.fromisoformat(slot["end"]) for slot in proposed_slots)

    busy_periods = get_busy_periods(service, start_datetime, end_datetime)

    free_slots = get_free_slots(busy_periods, proposed_slots)

    return {
        "slot_available": False,
        "free_slots": free_slots,
        "selected_slot": {}
    }
    
def router_function(state):
    if(state["slot_available"] == True):
        return "available"
    else:
        return "unavailable"


def free_slots_router(state):
    if(len(state["free_slots"]) == 0):
        return "no_slots"
    else:
        return "slots_found"
    
    
workflow = StateGraph(RecruitmentState)

workflow.add_node("prepare_candidate", prepare_candidate)
workflow.add_node("find_available_slots", find_available_slots)
workflow.add_node("draft_interview_email", draft_interview_email)
workflow.add_node("send_out_email", send_out_email)
workflow.add_node("select_slot", select_slot)
workflow.add_node("add_slot_to_calendar", add_slot_to_calendar)
workflow.add_node("recheck_selected_slot",recheck_selected_slot)


workflow.add_edge(START, "prepare_candidate")
workflow.add_edge("prepare_candidate", "find_available_slots")
workflow.add_conditional_edges("find_available_slots" , free_slots_router , {"slots_found" : "draft_interview_email" , "no_slots" : END})
workflow.add_edge("draft_interview_email", "send_out_email")
workflow.add_edge("send_out_email", "select_slot")
workflow.add_edge("select_slot", "recheck_selected_slot")
workflow.add_conditional_edges("recheck_selected_slot" , router_function , {"available" : "add_slot_to_calendar" , "unavailable" : "select_slot"})
workflow.add_edge("add_slot_to_calendar", END)

connection = sqlite3.connect("recruitment_checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(connection)

app = workflow.compile(checkpointer=memory)


def start_candidate_workflows(top_candidates, proposed_slots):

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, candidate_data in enumerate(top_candidates):

        thread_id = f"candidate_{i + 1}_{run_stamp}"

        initial_state = {
            "candidate": candidate_data,
            "free_slots": [],
            "proposed_slots": proposed_slots,
            "selected_slot": {},
            "email_draft": {},
            "thread_id": thread_id,
            "slot_available": False
        }

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # Same reasoning as the screening loop in main.py. A refused SMTP login
        # or a stale calendar token on the first candidate would otherwise mean
        # the second and third are never written to at all, and the run looks
        # like it finished. By this point every score and report is already on
        # disk, so one failed invitation is worth reporting and carrying on
        # from rather than losing the other candidates over.
        try:
            app.invoke(initial_state, config=config)

        except Exception as error:
            print(
                f"  could not start the workflow for "
                f"{candidate_data.get('name')} <{candidate_data.get('email')}>: "
                f"{type(error).__name__}: {error}"
            )