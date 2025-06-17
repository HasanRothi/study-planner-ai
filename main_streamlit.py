import streamlit as st
from main import StudyPlannerAI
import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')



def main_ui():
    st.set_page_config(
        page_title="Smart AI Study Planner",
        page_icon="📚",
        layout="centered"
    )

    st.title("📚 Smart AI Study Planner")
    st.markdown("""
    Welcome to your AI-powered study assistant! This tool helps you create an optimized study schedule 
    by prioritizing subjects where you are weakest.
    """)

    # Initialize StudyPlannerAI in Streamlit session state
    if 'planner' not in st.session_state:
        st.session_state.planner = StudyPlannerAI()
        st.session_state.planner.load_data()  # Try to load data on startup
        st.session_state.planner.train_model()  # Train model once on startup

    planner = st.session_state.planner

    st.sidebar.header("Settings & Actions")

    # --- Set Daily Study Hours ---
    st.sidebar.subheader("1. Set Daily Study Hours")
    current_hours = planner.total_daily_hours
    new_hours = st.sidebar.number_input(
        f"Total Daily Study Hours (current: {current_hours}h)",
        min_value=1.0,
        max_value=16.0,
        value=float(current_hours),
        step=0.5,
        format="%.1f",
        key="daily_hours_input"
    )
    if new_hours != current_hours:
        if planner.set_daily_hours(new_hours):
            st.sidebar.success(f"Daily study hours set to {new_hours}h!")
        else:
            st.sidebar.error("Invalid daily hours. Must be between 1-16.")

    st.sidebar.markdown("---")

    # --- Add Subject ---
    st.sidebar.subheader("2. Add New Subject")
    with st.sidebar.form("add_subject_form", clear_on_submit=True):
        subject_name = st.text_input("Subject Name")
        weakness_score = st.slider("Weakness Score (1=Strong, 10=Weak)", 1, 10, 5)
        difficulty_preference = st.slider("Difficulty Preference (0=Easy, 1=Hard)", 0.0, 1.0, 0.5, step=0.1)

        submitted = st.form_submit_button("Add Subject")
        if submitted:
            if subject_name:
                subject = planner.add_subject(subject_name, weakness_score, difficulty_preference)
                if subject:
                    st.sidebar.success(
                        f"Added {subject_name} (Weakness: {weakness_score}/10, Allocated: {subject['time_allocation_ratio']:.2f})")
                else:
                    st.sidebar.error("Failed to add subject. Check inputs.")
            else:
                st.sidebar.error("Subject name cannot be empty.")

    st.sidebar.markdown("---")

    # --- Save/Load Data ---
    st.sidebar.subheader("3. Data Management")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("💾 Save Data", key="save_data_btn"):
            planner.save_data()
            st.sidebar.success("Data saved successfully!")
    with col2:
        if st.button("📂 Load Data", key="load_data_btn"):
            if planner.load_data():
                st.sidebar.success("Data loaded successfully!")
            else:
                st.sidebar.warning("No saved data found or error loading.")

    st.sidebar.markdown("---")

    # --- Main Content Area ---
    st.header("Your Subjects")
    if planner.subjects:
        subjects_df = st.dataframe(
            [
                {
                    "ID": s['id'],
                    "Subject": s['name'],
                    "Weakness Score": f"{s['weakness_score']}/10",
                    "Difficulty Pref.": f"{s['difficulty_preference']:.1f}",
                    "AI Allocation Ratio": f"{s['time_allocation_ratio']:.3f}"
                }
                for s in planner.subjects
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No subjects added yet. Add some subjects from the sidebar!")

    st.markdown("---")

    # --- Generate Schedule ---
    st.header("🗓️ Generate Study Schedule")
    col_start, col_end = st.columns(2)
    with col_start:
        start_time = st.time_input("Start Time", value=planner._parse_time("09:00"), key="start_time_input")
    with col_end:
        end_time = st.time_input("End Time", value=planner._parse_time("21:00"), key="end_time_input")

    if st.button("🚀 Generate Schedule", type="primary"):
        if not planner.subjects:
            st.warning("Please add subjects before generating a schedule.")
        else:
            with st.spinner("Generating AI-optimized schedule..."):
                schedule = planner.generate_study_schedule(
                    start_time.strftime("%H:%M"),
                    end_time.strftime("%H:%M")
                )
                if schedule:
                    st.session_state.schedule = schedule
                    st.success("Schedule generated successfully!")
                    st.balloons()
                else:
                    st.error("Could not generate schedule. Please check your total daily hours and time window.")

    st.markdown("---")

    # --- Display Schedule ---
    st.header("🎯 Your Optimized Study Schedule")
    if 'schedule' in st.session_state and st.session_state.schedule:
        schedule = st.session_state.schedule

        # Display time distribution summary
        st.subheader(f"📊 Time Distribution (Total Daily: {planner.total_daily_hours}h)")
        time_dist = planner.calculate_time_distribution()
        if time_dist:
            time_dist_data = []
            for subject_name, data in time_dist.items():
                time_dist_data.append({
                    "Subject": subject_name,
                    "Hours": f"{data['hours']:.1f}h",
                    "Percentage": f"{data['percentage']:.1f}%"
                })
            st.dataframe(time_dist_data, use_container_width=True, hide_index=True)

        st.subheader("Detailed Schedule")
        for i, session in enumerate(schedule):
            st.markdown(f"#### {i + 1}. {session.get('time', 'N/A')} - {session.get('end_time', 'N/A')}")
            st.markdown(f"**Subject:** {session['subject']}")
            st.markdown(f"**Duration:** {session['duration_display']}")
            st.markdown(f"**Type:** {session['type']} (Intensity: {session['intensity']})")
            if 'weakness_score' in session:
                st.markdown(f"**Weakness:** {session['weakness_score']}/10")
            st.markdown("---")
    else:
        st.info("Generate a schedule to see it here!")

    st.markdown("---")

    # --- Analytics ---
    st.header("📈 AI Analytics & Insights")
    if st.button("View Analytics"):
        if planner.subjects:
            # Capture print output from get_analytics
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                planner.get_analytics()
            s = f.getvalue()

            # Parse and display
            lines = s.split('\n')

            st.subheader(lines[2].strip())  # AI ANALYTICS & INSIGHTS header
            st.write(lines[4].strip())  # DAILY STUDY TIME
            st.write(lines[5].strip())  # TOTAL SUBJECTS
            st.write(lines[6].strip())  # MINIMUM SESSION TIME

            # Split into categories
            current_category = None
            category_data = []

            for line in lines[7:]:  # Start from where categories begin
                line = line.strip()
                if line.startswith("🔴 WEAK SUBJECTS"):
                    if category_data: st.markdown("".join(category_data)); category_data = []
                    current_category = "weak"
                    st.subheader(line)
                elif line.startswith("🟡 MEDIUM SUBJECTS"):
                    if category_data: st.markdown("".join(category_data)); category_data = []
                    current_category = "medium"
                    st.subheader(line)
                elif line.startswith("🟢 STRONG SUBJECTS"):
                    if category_data: st.markdown("".join(category_data)); category_data = []
                    current_category = "strong"
                    st.subheader(line)
                elif line.startswith("💡 AI RECOMMENDATIONS"):
                    if category_data: st.markdown("".join(category_data)); category_data = []
                    current_category = "recommendations"
                    st.subheader(line)
                elif line.startswith("•"):  # Subject details
                    st.markdown(f"- {line[2:]}")  # Remove "• "
                elif current_category == "recommendations" and line:
                    st.markdown(f"- {line.strip()}")
                elif line:  # Any other significant line
                    if line != "=" * 60:  # Avoid printing the separator again
                        st.write(line)

            # Print any remaining category data
            if category_data:
                st.markdown("".join(category_data))
        else:
            st.info("Add subjects to view analytics.")


# Run the UI
if __name__ == "__main__":
    main_ui()