import streamlit as st

from datetime import datetime
from langgraph.types import Command

from agent import app


st.title("Interview Slot Selection")

thread_id = st.query_params.get("thread_id")

if thread_id is None:
    st.error("Invalid interview link")
    st.stop()

config = {
    "configurable": {
        "thread_id": thread_id
    }
}

current_state = app.get_state(config)

if not current_state.values:
    st.error("Interview session not found")
    st.stop()

if current_state.values.get("selected_slot") and len(current_state.next) == 0:

    selected_slot = current_state.values["selected_slot"]

    start = datetime.fromisoformat(selected_slot["start"])
    end = datetime.fromisoformat(selected_slot["end"])

    st.success("Your interview has already been scheduled.")

    st.write(
        start.strftime("%d %b %Y, %I:%M %p"),
        "-",
        end.strftime("%I:%M %p")
    )

    st.stop()

free_slots = current_state.values.get("free_slots", [])

if len(free_slots) == 0:
    st.error("No interview slots are currently available.")
    st.stop()

slot_options = []

for slot in free_slots:

    start = datetime.fromisoformat(slot["start"])
    end = datetime.fromisoformat(slot["end"])

    slot_text = f"{start.strftime('%d %b %Y, %I:%M %p')} - {end.strftime('%I:%M %p')}"

    slot_options.append(slot_text)

selected_option = st.radio("Choose your preferred interview slot:", slot_options)

if st.button("Confirm Slot"):

    chosen_slot_number = slot_options.index(selected_option) + 1

    result = app.invoke(Command(resume=chosen_slot_number), config=config)

    if "__interrupt__" in result and len(result["__interrupt__"]) > 0:
        st.warning("That slot was just taken by another candidate. Please choose another available slot.")
        st.rerun()

    else:
        st.success("Your interview slot has been booked successfully.")