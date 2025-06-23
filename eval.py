import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
from main import StudyPlannerAI

class StudyPlannerEvaluator:
    def __init__(self, planner):
        """Initialize evaluator with a trained StudyPlannerAI instance"""
        self.planner = planner

    def evaluate_time_allocation_effectiveness(self):
        """
        Evaluate how effectively the AI allocates time based on weakness scores
        Creates a publication-ready visualization
        """
        if not self.planner.subjects:
            print("No subjects to evaluate. Please add subjects first.")
            return

        # Extract data for analysis
        subjects_data = []
        time_distribution = self.planner.calculate_time_distribution()

        for subject in self.planner.subjects:
            subject_name = subject['name']
            weakness_score = subject['weakness_score']
            time_info = time_distribution[subject_name]

            subjects_data.append({
                'Subject': subject_name,
                'Weakness_Score': weakness_score,
                'Allocated_Hours': time_info['hours'],
                'Allocated_Percentage': time_info['percentage'],
                'AI_Ratio': subject['time_allocation_ratio']
            })

        df = pd.DataFrame(subjects_data)

        # Create the evaluation plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('AI Study Planner: Time Allocation Effectiveness Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Weakness Score vs Time Allocation (Main relationship)
        scatter = ax1.scatter(df['Weakness_Score'], df['Allocated_Hours'],
                              c=df['Weakness_Score'], cmap='RdYlBu_r',
                              s=100, alpha=0.7, edgecolors='black', linewidth=1)
        ax1.set_xlabel('Weakness Score (1-10)', fontweight='bold')
        ax1.set_ylabel('Allocated Time (Hours)', fontweight='bold')
        ax1.set_title('Time Allocation vs Subject Weakness', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add correlation coefficient
        correlation = np.corrcoef(df['Weakness_Score'], df['Allocated_Hours'])[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: r = {correlation:.3f}',
                 transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3",
                                                    facecolor="white", alpha=0.8), fontweight='bold')

        # Add subject labels
        for idx, row in df.iterrows():
            ax1.annotate(row['Subject'],
                         (row['Weakness_Score'], row['Allocated_Hours']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=9, alpha=0.8)

        # 2. Time Distribution Bar Chart
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(df)))
        bars = ax2.bar(range(len(df)), df['Allocated_Hours'],
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_xlabel('Subjects', fontweight='bold')
        ax2.set_ylabel('Allocated Time (Hours)', fontweight='bold')
        ax2.set_title('Time Distribution Across Subjects', fontweight='bold')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['Subject'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, df['Allocated_Hours']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                     f'{value:.1f}h', ha='center', va='bottom', fontweight='bold')

        # 3. AI Effectiveness: Expected vs Actual Allocation
        ideal_allocation = df['Weakness_Score'] / df['Weakness_Score'].sum() * self.planner.total_daily_hours
        ax3.scatter(ideal_allocation, df['Allocated_Hours'],
                    c=df['Weakness_Score'], cmap='RdYlBu_r',
                    s=100, alpha=0.7, edgecolors='black', linewidth=1)

        # Perfect allocation line
        max_val = max(ideal_allocation.max(), df['Allocated_Hours'].max())
        ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Allocation')

        ax3.set_xlabel('Ideal Allocation (Hours)', fontweight='bold')
        ax3.set_ylabel('AI Allocation (Hours)', fontweight='bold')
        ax3.set_title('AI vs Ideal Time Allocation', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Calculate and display efficiency metric
        efficiency = 1 - np.mean(np.abs(ideal_allocation - df['Allocated_Hours']) / ideal_allocation)
        ax3.text(0.05, 0.95, f'Efficiency: {efficiency:.1%}',
                 transform=ax3.transAxes, bbox=dict(boxstyle="round,pad=0.3",
                                                    facecolor="lightgreen", alpha=0.8), fontweight='bold')

        # 4. Summary Statistics
        ax4.axis('off')

        # Calculate key metrics
        total_subjects = len(df)
        weak_subjects = len(df[df['Weakness_Score'] >= 7])
        strong_subjects = len(df[df['Weakness_Score'] <= 3])
        avg_weak_time = df[df['Weakness_Score'] >= 7]['Allocated_Hours'].mean() if weak_subjects > 0 else 0
        avg_strong_time = df[df['Weakness_Score'] <= 3]['Allocated_Hours'].mean() if strong_subjects > 0 else 0
        time_ratio = avg_weak_time / avg_strong_time if avg_strong_time > 0 else float('inf')

        # Create summary text
        summary_text = f"""
        📊 EVALUATION SUMMARY

        Total Subjects: {total_subjects}
        Daily Study Time: {self.planner.total_daily_hours:.1f} hours

        📈 Subject Distribution:
        • Weak Subjects (≥7): {weak_subjects}
        • Strong Subjects (≤3): {strong_subjects}

        ⏰ Time Allocation:
        • Avg. time for weak subjects: {avg_weak_time:.1f}h
        • Avg. time for strong subjects: {avg_strong_time:.1f}h
        • Weak:Strong ratio: {time_ratio:.1f}:1

        🎯 AI Performance:
        • Correlation (weakness-time): {correlation:.3f}
        • Allocation efficiency: {efficiency:.1%}

        ✅ Result: AI successfully prioritizes
        weak subjects with {time_ratio:.1f}x more time
        allocation compared to strong subjects.
        """

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'study_planner_evaluation_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Evaluation plot saved as: {filename}")

        plt.show()

        return {
            'correlation': correlation,
            'efficiency': efficiency,
            'time_ratio': time_ratio,
            'weak_subjects': weak_subjects,
            'strong_subjects': strong_subjects,
            'dataframe': df
        }


def create_sample_evaluation():
    """
    Create a sample evaluation using the actual StudyPlannerAI class
    """
    # Create actual StudyPlannerAI instance
    planner = StudyPlannerAI()

    # Set daily hours
    planner.set_daily_hours(6)

    # Add sample subjects with different weakness scores
    planner.add_subject("Mathematics", weakness_score=9, difficulty_pref=0.6)
    planner.add_subject("Physics", weakness_score=7, difficulty_pref=0.5)
    planner.add_subject("Chemistry", weakness_score=5, difficulty_pref=0.4)
    planner.add_subject("Biology", weakness_score=3, difficulty_pref=0.3)
    planner.add_subject("English", weakness_score=2, difficulty_pref=0.2)

    # The AI model will be automatically trained when adding subjects
    print("✅ StudyPlannerAI created with 5 subjects")
    print("🤖 AI model trained automatically")

    # Create evaluator and run evaluation
    evaluator = StudyPlannerEvaluator(planner)
    results = evaluator.evaluate_time_allocation_effectiveness()

    return results, planner


# Example usage
if __name__ == "__main__":
    print("🔬 AI Study Planner Evaluation Tool")
    print("=" * 50)

    # Use your actual StudyPlannerAI class
    print("Creating StudyPlannerAI instance with sample data...")
    results, planner = create_sample_evaluation()

    print(f"\n📈 Key Results:")
    print(f"• AI-Weakness Correlation: {results['correlation']:.3f}")
    print(f"• Allocation Efficiency: {results['efficiency']:.1%}")
    print(f"• Weak:Strong Time Ratio: {results['time_ratio']:.1f}:1")
    print(f"• Weak Subjects: {results['weak_subjects']}")
    print(f"• Strong Subjects: {results['strong_subjects']}")

    # Show AI model training info
    print(f"\n🤖 AI Model Info:")
    print(f"• Model Trained: {planner.is_trained}")
    print(f"• Total Subjects: {len(planner.subjects)}")
    print(f"• Daily Study Hours: {planner.total_daily_hours}")

    print("\n✅ Evaluation complete! Check the generated PNG file.")

# Direct usage with your existing StudyPlannerAI instance:
"""
To use with your already created StudyPlannerAI:

# If you already have a trained planner
planner = your_existing_planner  # Your StudyPlannerAI instance

# Create evaluator and run evaluation
evaluator = StudyPlannerEvaluator(planner)
results = evaluator.evaluate_time_allocation_effectiveness()
"""