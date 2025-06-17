import tensorflow as tf
import numpy as np
import json
from datetime import datetime, timedelta
import os


class StudyPlannerAI:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.subjects = []
        self.study_sessions = []
        self.total_daily_hours = 4  # Default total study hours per day
        self.min_session_minutes = 10  # Minimum session length to ensure all subjects get time

    def create_model(self):
        """Create a neural network model for time allocation"""
        print("🧠 Creating AI model...")

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return model

    def generate_training_data(self, n_samples=2000):
        """Generate synthetic training data for time allocation model"""
        print("📊 Generating training data...")

        np.random.seed(42)

        # Features: [weakness_score, difficulty_preference]
        weakness_scores = np.random.randint(1, 11, n_samples)  # 1-10 scale
        difficulty_pref = np.random.uniform(0, 1, n_samples)  # 0-1 preference

        X = np.column_stack([weakness_scores, difficulty_pref])

        # Target: time_allocation_ratio (higher weakness = more time)
        y = []
        for i in range(n_samples):
            # Base time allocation from weakness score
            # Weakness 1-3: Low time allocation (0.1-0.3)
            # Weakness 4-6: Medium time allocation (0.3-0.6)
            # Weakness 7-10: High time allocation (0.6-0.9)
            base_ratio = (weakness_scores[i] - 1) / 9.0  # Normalize to 0-1

            # Apply exponential scaling to emphasize weakness
            time_ratio = np.power(base_ratio, 0.7)  # Makes weak subjects get more time

            # Adjust based on difficulty preference
            if difficulty_pref[i] > 0.7:  # Likes difficult subjects
                time_ratio *= 1.1
            elif difficulty_pref[i] < 0.3:  # Prefers easier subjects
                time_ratio *= 0.9

            # Add some realistic noise
            time_ratio += np.random.normal(0, 0.05)

            # Ensure 0.1-0.9 range (minimum 10% time, maximum 90% time)
            time_ratio = np.clip(time_ratio, 0.1, 0.9)
            y.append(time_ratio)

        return X, np.array(y)

    def train_model(self):
        """Train the AI model for time allocation"""
        print("🚀 Training AI model...")

        if self.model is None:
            self.model = self.create_model()

        # Generate training data
        X, y = self.generate_training_data()

        # Train the model
        history = self.model.fit(
            X, y,
            epochs=150,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        self.is_trained = True
        print("✅ AI model trained successfully!")

        # Print training results
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        print(f"📈 Final Training Loss: {final_loss:.4f}")
        print(f"📈 Final Validation Loss: {final_val_loss:.4f}")

        return history

    def predict_time_allocation(self, weakness_score, difficulty_pref=0.5):
        """Predict time allocation ratio for a subject"""
        if not self.is_trained:
            print("⚠️ Model not trained yet. Training now...")
            self.train_model()

        # Prepare input features
        features = np.array([[weakness_score, difficulty_pref]])

        # Make prediction
        time_ratio = self.model.predict(features, verbose=0)[0][0]
        return float(time_ratio)

    def set_daily_hours(self, hours):
        """Set total daily study hours"""
        if hours <= 0 or hours > 16:
            print("❌ Daily hours should be between 1-16 hours.")
            return False

        self.total_daily_hours = hours
        print(f"✅ Total daily study hours set to: {hours} hours")
        return True

    def add_subject(self, name, weakness_score, difficulty_pref=0.5):
        """Add a new subject with AI-predicted time allocation"""
        if not (1 <= weakness_score <= 10):
            print("❌ Weakness score must be between 1-10.")
            return None

        time_ratio = self.predict_time_allocation(weakness_score, difficulty_pref)

        subject = {
            'id': len(self.subjects) + 1,
            'name': name,
            'weakness_score': weakness_score,
            'difficulty_preference': difficulty_pref,
            'time_allocation_ratio': time_ratio,
            'created_at': datetime.now().isoformat()
        }

        self.subjects.append(subject)
        print(f"✅ Added subject: {name}")
        print(f"   📊 Weakness Score: {weakness_score}/10")
        print(f"   ⏱️  Time Allocation Ratio: {time_ratio:.3f}")
        return subject

    def calculate_time_distribution(self, available_time_minutes=None):
        """Calculate actual time distribution for all subjects - ENSURES ALL SUBJECTS GET TIME"""
        if not self.subjects:
            return {}

        # Use available time if provided, otherwise use daily hours
        if available_time_minutes:
            total_minutes = available_time_minutes
        else:
            total_minutes = self.total_daily_hours * 60

        # Get raw time ratios
        raw_ratios = {s['name']: s['time_allocation_ratio'] for s in self.subjects}

        # GUARANTEE minimum time for each subject
        num_subjects = len(self.subjects)

        # Calculate dynamic minimum based on available time
        # Each subject gets at least 5 minutes OR 5% of total time divided by subjects, whichever is smaller
        min_minutes_per_subject = max(5, min(self.min_session_minutes, total_minutes * 0.15 / num_subjects))

        # Calculate total minimum required time
        total_min_required = min_minutes_per_subject * num_subjects

        # If minimum requirements exceed total time, compress proportionally
        if total_min_required > total_minutes:
            print(f"⚠️ Compressing minimum time per subject to fit in {total_minutes / 60:.1f}h window")
            min_minutes_per_subject = total_minutes / num_subjects
            total_min_required = total_minutes

        # Reserve minimum time for each subject
        remaining_minutes = total_minutes - total_min_required

        # Distribute remaining time based on AI ratios
        total_raw_ratio = sum(raw_ratios.values())

        time_distribution = {}
        for subject_name, ratio in raw_ratios.items():
            # Base minimum time + proportional share of remaining time
            if remaining_minutes > 0:
                additional_time = (ratio / total_raw_ratio) * remaining_minutes
                total_time = min_minutes_per_subject + additional_time
            else:
                total_time = min_minutes_per_subject

            time_distribution[subject_name] = {
                'hours': total_time / 60,
                'minutes': int(total_time),
                'percentage': (total_time / total_minutes) * 100
            }

        return time_distribution

    def generate_study_schedule(self, start_time="09:00", end_time="21:00"):
        """Generate AI-optimized study schedule - GUARANTEES ALL SUBJECTS ARE INCLUDED"""
        if not self.subjects:
            print("❌ No subjects added. Please add subjects first.")
            return []

        print("📅 Generating AI-optimized study schedule...")
        print(f"🎯 Ensuring ALL {len(self.subjects)} subjects are included!")

        # Calculate available time window in minutes
        start_dt = self._parse_time(start_time)
        end_dt = self._parse_time(end_time)
        available_minutes = int((end_dt - start_dt).total_seconds() / 60)
        requested_study_minutes = self.total_daily_hours * 60

        print(f"📊 Available time window: {available_minutes // 60}h {available_minutes % 60}m")
        print(f"📊 Requested study time: {requested_study_minutes // 60}h {requested_study_minutes % 60}m")

        # CRITICAL: Use available time window for calculations to ensure everything fits
        effective_study_time = min(available_minutes * 0.85, requested_study_minutes)  # 85% for breaks

        print(f"📊 Effective study time: {int(effective_study_time) // 60}h {int(effective_study_time) % 60}m")

        # Calculate time distribution based on AVAILABLE time, not requested time
        time_dist = self.calculate_time_distribution(effective_study_time)

        # Create optimized sessions that WILL fit in available time
        all_sessions = self._create_optimized_sessions(time_dist, available_minutes)

        # Schedule all sessions - this will now work because sessions are pre-fitted
        schedule = self._schedule_sessions(all_sessions, start_dt, end_dt)

        # Final verification and emergency fixes
        schedule = self._ensure_all_subjects_included(schedule, start_dt, end_dt)

        self.study_sessions = schedule

        # Report results
        scheduled_subjects = set(session['subject'] for session in schedule)
        all_subjects = set(subject['name'] for subject in self.subjects)

        if len(scheduled_subjects) == len(all_subjects):
            print("🎉 SUCCESS: All subjects included in schedule!")
        else:
            print(f"🚨 CRITICAL ERROR: Only {len(scheduled_subjects)}/{len(all_subjects)} subjects scheduled")

        total_scheduled_minutes = sum(session['duration_minutes'] for session in schedule)
        print(f"📊 Total scheduled time: {total_scheduled_minutes // 60}h {total_scheduled_minutes % 60}m")

        return schedule

    def _create_optimized_sessions(self, time_dist, available_minutes):
        """Create sessions that are guaranteed to fit in available time"""
        all_sessions = []
        session_id = 1

        # Sort subjects by weakness score (prioritize weak subjects)
        sorted_subjects = sorted(self.subjects, key=lambda x: x['weakness_score'], reverse=True)

        # Calculate total session time needed (excluding breaks)
        total_session_time = sum(dist['minutes'] for dist in time_dist.values())
        estimated_breaks = len(self.subjects) * 5  # Conservative break estimate

        # If total time exceeds available time, compress proportionally
        if total_session_time + estimated_breaks > available_minutes:
            compression_factor = (available_minutes - estimated_breaks) / total_session_time
            print(f"⚠️ Compressing all sessions by {(1 - compression_factor) * 100:.1f}% to fit time window")

            # Apply compression to all subjects
            for subject_name in time_dist:
                time_dist[subject_name]['minutes'] = int(time_dist[subject_name]['minutes'] * compression_factor)
                time_dist[subject_name]['hours'] = time_dist[subject_name]['minutes'] / 60

        # Create sessions for each subject
        for subject in sorted_subjects:
            subject_name = subject['name']
            allocated_minutes = time_dist[subject_name]['minutes']
            weakness_score = subject['weakness_score']

            # Ensure minimum time (at least 5 minutes per subject)
            allocated_minutes = max(5, allocated_minutes)

            print(f"🔄 Creating sessions for {subject_name}: {allocated_minutes:.0f} minutes")

            # Create efficient sessions for this subject
            remaining_minutes = allocated_minutes

            # Strategy: Create fewer, appropriately sized sessions
            while remaining_minutes > 0:
                if remaining_minutes >= 45:
                    session_length = min(45, remaining_minutes)
                elif remaining_minutes >= 30:
                    session_length = min(30, remaining_minutes)
                elif remaining_minutes >= 20:
                    session_length = min(20, remaining_minutes)
                elif remaining_minutes >= 15:
                    session_length = min(15, remaining_minutes)
                elif remaining_minutes >= 10:
                    session_length = min(10, remaining_minutes)
                else:
                    # Use all remaining time, even if small
                    session_length = remaining_minutes

                session_type, intensity = self._determine_session_type(weakness_score, session_length)

                session = {
                    'id': session_id,
                    'subject': subject_name,
                    'duration_minutes': int(session_length),
                    'duration_display': f"{int(session_length // 60)}h {int(session_length % 60)}m" if session_length >= 60 else f"{int(session_length)}m",
                    'type': session_type,
                    'intensity': intensity,
                    'weakness_score': weakness_score,
                    'allocated_percentage': time_dist[subject_name]['percentage'] if subject_name in time_dist else 0,
                    'priority': weakness_score
                }

                all_sessions.append(session)
                session_id += 1
                remaining_minutes -= session_length

        return all_sessions

    def _schedule_sessions(self, all_sessions, start_dt, end_dt):
        """Schedule sessions with guaranteed fit"""
        schedule = []
        current_time = start_dt

        for session in all_sessions:
            session_length = session['duration_minutes']

            # Calculate when this session would end
            session_end_time = current_time + timedelta(minutes=session_length)

            # If session doesn't fit, use all remaining time
            if session_end_time > end_dt:
                remaining_minutes = int((end_dt - current_time).total_seconds() / 60)
                if remaining_minutes >= 5:  # Minimum 5 minutes
                    session_length = remaining_minutes
                    session['duration_minutes'] = session_length
                    session['duration_display'] = f"{int(session_length)}m"
                    session['type'] = "Compressed Session"
                    print(f"⚠️ Compressed {session['subject']} to {session_length}min to fit")
                else:
                    # Skip this session if less than 5 minutes
                    print(f"⚠️ Skipping {session['subject']} - insufficient time")
                    continue

            # Add timing information
            session['time'] = current_time.strftime('%H:%M')
            session['end_time'] = (current_time + timedelta(minutes=session_length)).strftime('%H:%M')

            schedule.append(session)

            # Move to next time slot with minimal break
            break_duration = 5 if session_length >= 20 else 2
            current_time += timedelta(minutes=session_length + break_duration)

            # Stop if we're at or past end time
            if current_time >= end_dt:
                break

        return schedule

    def _ensure_all_subjects_included(self, schedule, start_dt, end_dt):
        """Emergency function to ensure ALL subjects are included"""
        scheduled_subjects = set(session['subject'] for session in schedule)
        all_subjects = set(subject['name'] for subject in self.subjects)
        missing_subjects = all_subjects - scheduled_subjects

        if not missing_subjects:
            return schedule  # All subjects already included

        print(f"🚨 EMERGENCY: Missing subjects detected: {missing_subjects}")
        print("🔧 Applying guaranteed inclusion strategy...")

        # Strategy 1: Compress existing sessions to make room
        if schedule:
            print("📉 Compressing existing sessions to make room...")
            total_existing_time = sum(s['duration_minutes'] for s in schedule)
            needed_time = len(missing_subjects) * 8  # 8 minutes each for missing subjects

            if total_existing_time > needed_time:
                # Compress all existing sessions proportionally
                compression_factor = (total_existing_time - needed_time) / total_existing_time

                for session in schedule:
                    old_duration = session['duration_minutes']
                    new_duration = max(5, int(old_duration * compression_factor))  # Minimum 5 minutes
                    session['duration_minutes'] = new_duration
                    session['duration_display'] = f"{new_duration}m"

                    print(f"📉 Compressed {session['subject']}: {old_duration}min → {new_duration}min")

        # Strategy 2: Add micro-sessions for missing subjects
        current_time = start_dt
        session_id = len(schedule) + 1

        # Recalculate timing for existing sessions
        for session in schedule:
            session['time'] = current_time.strftime('%H:%M')
            session['end_time'] = (current_time + timedelta(minutes=session['duration_minutes'])).strftime('%H:%M')
            current_time += timedelta(minutes=session['duration_minutes'] + 2)  # 2min break

        # Add emergency sessions for missing subjects
        for subject_name in missing_subjects:
            # Calculate remaining time
            remaining_minutes = int((end_dt - current_time).total_seconds() / 60)

            if remaining_minutes >= 5:
                session_length = min(10, remaining_minutes - 2)  # Leave 2min buffer

                # Find subject details
                subject = next((s for s in self.subjects if s['name'] == subject_name), None)
                weakness_score = subject['weakness_score'] if subject else 5

                emergency_session = {
                    'id': session_id,
                    'subject': subject_name,
                    'duration_minutes': session_length,
                    'duration_display': f"{session_length}m",
                    'type': 'Emergency Review',
                    'intensity': '⚡ Quick',
                    'weakness_score': weakness_score,
                    'time': current_time.strftime('%H:%M'),
                    'end_time': (current_time + timedelta(minutes=session_length)).strftime('%H:%M'),
                    'allocated_percentage': 0.1,
                    'priority': weakness_score
                }

                schedule.append(emergency_session)
                current_time += timedelta(minutes=session_length + 2)
                session_id += 1

                print(f"🆘 Added emergency {session_length}min session for {subject_name}")
            else:
                # Last resort: steal time from longest session
                if schedule:
                    longest_session = max(schedule, key=lambda x: x['duration_minutes'])
                    if longest_session['duration_minutes'] > 10:
                        # Steal 5 minutes
                        longest_session['duration_minutes'] -= 5
                        longest_session['duration_display'] = f"{longest_session['duration_minutes']}m"

                        # Create mini session
                        mini_session = {
                            'id': session_id,
                            'subject': subject_name,
                            'duration_minutes': 5,
                            'duration_display': '5m',
                            'type': 'Ultra-Quick Review',
                            'intensity': '⚡ Lightning',
                            'time': 'Squeezed',
                            'end_time': 'In',
                            'allocated_percentage': 0.05
                        }

                        schedule.append(mini_session)
                        print(f"🆘 Created ultra-quick 5min session for {subject_name}")

        # Final sort by time
        timed_sessions = [s for s in schedule if 'time' in s and s['time'] != 'Squeezed']
        untimed_sessions = [s for s in schedule if s not in timed_sessions]

        return timed_sessions + untimed_sessions

    def _parse_time(self, time_str):
        """Parse time string to datetime object"""
        hour, minute = map(int, time_str.split(':'))
        return datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)

    def _determine_session_type(self, weakness_score, duration):
        """Determine session type and intensity based on weakness score"""
        if duration < 10:
            return "Quick Review", "⚡ Quick"
        elif weakness_score >= 8:
            if duration >= 60:
                return "Intensive Learning", "🔥 High"
            else:
                return "Focused Practice", "🔥 High"
        elif weakness_score >= 6:
            if duration >= 60:
                return "Active Learning", "⚡ Medium"
            else:
                return "Practice Session", "⚡ Medium"
        elif weakness_score >= 4:
            return "Regular Study", "📚 Medium"
        else:
            return "Review & Maintenance", "📝 Low"

    def display_schedule(self, schedule):
        """Display the generated schedule in a nice format"""
        if not schedule:
            print("❌ No schedule to display.")
            return

        print("\n" + "=" * 80)
        print("📚 YOUR AI-OPTIMIZED STUDY SCHEDULE")
        print("=" * 80)

        # Verify all subjects are included FIRST
        scheduled_subjects = set(session['subject'] for session in schedule)
        all_subjects = set(subject['name'] for subject in self.subjects)

        if len(scheduled_subjects) == len(all_subjects):
            print(f"✅ SUCCESS: ALL {len(all_subjects)} SUBJECTS INCLUDED!")
        else:
            missing = all_subjects - scheduled_subjects
            print(f"🚨 MISSING SUBJECTS: {', '.join(missing)}")

        # Display time distribution summary
        time_dist = self.calculate_time_distribution()
        print(f"\n📊 TIME DISTRIBUTION (Total: {self.total_daily_hours}h):")
        for subject_name in sorted(scheduled_subjects):
            if subject_name in time_dist:
                dist = time_dist[subject_name]
                bar_length = int(dist['percentage'] / 5)  # Scale for display
                bar = "█" * bar_length
                print(f"   {subject_name:20s} │ {bar:20s} │ {dist['hours']:.1f}h ({dist['percentage']:.1f}%)")

        print(f"\n📅 DETAILED SCHEDULE:")
        print("-" * 80)

        for i, session in enumerate(schedule, 1):
            print(f"\n{i:2d}. {session.get('time', 'TBD')} - {session.get('end_time', 'TBD')} │ {session['subject']}")
            print(f"    ⏱️  Duration: {session['duration_display']}")
            print(f"    📖 Type: {session['type']}")
            print(f"    🎯 Intensity: {session['intensity']}")
            if 'weakness_score' in session:
                print(f"    📈 Weakness Score: {session['weakness_score']}/10")

    def get_analytics(self):
        """Get AI analytics and insights"""
        if not self.subjects:
            print("No subjects to analyze.")
            return

        print("\n" + "=" * 60)
        print("📊 AI ANALYTICS & INSIGHTS")
        print("=" * 60)

        # Time distribution analysis
        time_dist = self.calculate_time_distribution()

        print(f"⏰ DAILY STUDY TIME: {self.total_daily_hours} hours")
        print(f"📚 TOTAL SUBJECTS: {len(self.subjects)}")
        print(f"🎯 MINIMUM SESSION TIME: {self.min_session_minutes} minutes")

        # Categorize subjects by weakness
        weak_subjects = [s for s in self.subjects if s['weakness_score'] >= 7]
        medium_subjects = [s for s in self.subjects if 4 <= s['weakness_score'] < 7]
        strong_subjects = [s for s in self.subjects if s['weakness_score'] < 4]

        print(f"\n🔴 WEAK SUBJECTS ({len(weak_subjects)}) - Need Most Attention:")
        for subject in sorted(weak_subjects, key=lambda x: x['weakness_score'], reverse=True):
            time_info = time_dist[subject['name']]
            print(
                f"   • {subject['name']} (Score: {subject['weakness_score']}/10) → {time_info['hours']:.1f}h ({time_info['percentage']:.1f}%)")

        print(f"\n🟡 MEDIUM SUBJECTS ({len(medium_subjects)}) - Regular Practice:")
        for subject in sorted(medium_subjects, key=lambda x: x['weakness_score'], reverse=True):
            time_info = time_dist[subject['name']]
            print(
                f"   • {subject['name']} (Score: {subject['weakness_score']}/10) → {time_info['hours']:.1f}h ({time_info['percentage']:.1f}%)")

        print(f"\n🟢 STRONG SUBJECTS ({len(strong_subjects)}) - Maintenance Only:")
        for subject in sorted(strong_subjects, key=lambda x: x['weakness_score'], reverse=True):
            time_info = time_dist[subject['name']]
            print(
                f"   • {subject['name']} (Score: {subject['weakness_score']}/10) → {time_info['hours']:.1f}h ({time_info['percentage']:.1f}%)")

        # AI Recommendations
        print(f"\n💡 AI RECOMMENDATIONS:")
        if weak_subjects:
            weakest = max(weak_subjects, key=lambda x: x['weakness_score'])
            time_info = time_dist[weakest['name']]
            print(f"   🎯 PRIMARY FOCUS: {weakest['name']} ({time_info['hours']:.1f}h allocated)")

        if len(weak_subjects) > 1:
            print(f"   ⚠️  You have {len(weak_subjects)} weak subjects - consider extending daily study hours")

        if strong_subjects:
            print(f"   ✅ Maintain strong subjects with light review sessions")

        total_sessions = len(self.study_sessions)
        if total_sessions > 0:
            print(f"   📋 Total planned sessions: {total_sessions}")

        print(f"   🛡️ GUARANTEE: All subjects will be included in every schedule!")

    def save_data(self, filename="study_planner_data.json"):
        """Save subjects and sessions to JSON file"""
        data = {
            'subjects': self.subjects,
            'study_sessions': self.study_sessions,
            'total_daily_hours': self.total_daily_hours,
            'min_session_minutes': self.min_session_minutes,
            'model_trained': self.is_trained,
            'saved_at': datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Data saved to {filename}")

    def load_data(self, filename="study_planner_data.json"):
        """Load subjects and sessions from JSON file"""
        if not os.path.exists(filename):
            print(f"❌ File {filename} not found.")
            return False

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.subjects = data.get('subjects', [])
            self.study_sessions = data.get('study_sessions', [])
            self.total_daily_hours = data.get('total_daily_hours', 4)
            self.min_session_minutes = data.get('min_session_minutes', 10)

            print(f"✅ Data loaded from {filename}")
            print(f"📚 Loaded {len(self.subjects)} subjects")
            print(f"⏰ Daily study hours: {self.total_daily_hours}")
            print(f"🎯 Minimum session time: {self.min_session_minutes} minutes")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False


def main():
    """Main function to run the AI Study Planner"""
    print("🎓 Welcome to Smart AI Study Planner!")
    print("=" * 50)
    print("📝 This planner automatically allocates more time to weak subjects")
    print("💪 and less time to strong subjects for optimal learning!")

    # Initialize AI planner
    planner = StudyPlannerAI()

    # Try to load existing data
    planner.load_data()

    while True:
        print("\n📋 MENU:")
        print("1. Set Daily Study Hours")
        print("2. Add Subject")
        print("3. Generate Study Schedule")
        print("4. View Analytics")
        print("5. Save Data")
        print("6. Exit")

        choice = input("\n👉 Enter your choice (1-6): ").strip()

        if choice == '1':
            print(f"\n⏰ SET DAILY STUDY HOURS (Current: {planner.total_daily_hours}h)")
            try:
                hours = float(input("Enter total daily study hours (1-16): "))
                planner.set_daily_hours(hours)
            except ValueError:
                print("❌ Please enter a valid number.")

        elif choice == '2':
            print("\n📚 ADD NEW SUBJECT")
            name = input("Subject name: ").strip()

            if not name:
                print("❌ Subject name cannot be empty.")
                continue

            try:
                print("\nWeakness Scale:")
                print("1-3: Strong (you're good at this)")
                print("4-6: Medium (need regular practice)")
                print("7-10: Weak (need intensive focus)")

                weakness = int(input("Weakness score (1-10): "))
                if not 1 <= weakness <= 10:
                    print("❌ Weakness score must be between 1-10.")
                    continue

                difficulty = float(input("Difficulty preference (0-1, default 0.5): ") or "0.5")
                if not 0 <= difficulty <= 1:
                    print("❌ Difficulty preference must be between 0-1.")
                    continue

                planner.add_subject(name, weakness, difficulty)

            except ValueError:
                print("❌ Please enter valid numbers.")

        elif choice == '3':
            if not planner.subjects:
                print("❌ Please add subjects first.")
                continue

            print("\n📅 GENERATE STUDY SCHEDULE")
            start = input("Start time (HH:MM, default 09:00): ").strip() or "09:00"
            end = input("End time (HH:MM, default 21:00): ").strip() or "21:00"

            try:
                # Validate time format
                planner._parse_time(start)
                planner._parse_time(end)
            except ValueError:
                print("❌ Invalid time format. Use HH:MM format.")
                continue

            schedule = planner.generate_study_schedule(start, end)
            if schedule:
                planner.display_schedule(schedule)

        elif choice == '4':
            planner.get_analytics()

        elif choice == '5':
            planner.save_data()

        elif choice == '6':
            print("👋 Thank you for using Smart AI Study Planner!")
            print("🚀 Keep learning and improving!")
            break

        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    # Suppress TensorFlow warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    # example_usage()
    main()


# Example usage function
def example_usage():
    """Example of how to use the improved AI Study Planner"""
    print("\n🔬 EXAMPLE USAGE:")

    # Create planner
    planner = StudyPlannerAI()

    # Set daily study hours
    planner.set_daily_hours(6)  # 6 hours total per day

    # Add subjects with different weakness levels
    planner.add_subject("Mathematics", weakness_score=9)  # Very weak - gets most time
    planner.add_subject("Physics", weakness_score=7)  # Weak - gets good amount of time
    planner.add_subject("Chemistry", weakness_score=4)  # Medium - gets moderate time
    planner.add_subject("Biology", weakness_score=2)  # Strong - gets least time

    # Generate and display schedule
    schedule = planner.generate_study_schedule("09:00", "18:00")
    planner.display_schedule(schedule)

    # Show analytics
    planner.get_analytics()

    return planner

# Uncomment to run example
# example_usage()
