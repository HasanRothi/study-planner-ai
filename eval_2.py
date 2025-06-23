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

        # Create the evaluation plot with 2 subplots - adjusted spacing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('AI Study Planner: Time Allocation Effectiveness Analysis',
                     fontsize=16, fontweight='bold', y=0.95)

        # AI Effectiveness: Expected vs Actual Allocation
        ideal_allocation = df['Weakness_Score'] / df['Weakness_Score'].sum() * self.planner.total_daily_hours

        # Create scatter plot with better styling
        scatter = ax1.scatter(ideal_allocation, df['Allocated_Hours'],
                              c=df['Weakness_Score'], cmap='RdYlBu_r',
                              s=150, alpha=0.8, edgecolors='black', linewidth=1.5)

        # Perfect allocation line
        max_val = max(ideal_allocation.max(), df['Allocated_Hours'].max())
        min_val = min(ideal_allocation.min(), df['Allocated_Hours'].min())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7,
                 linewidth=2, label='Perfect Allocation Line')

        # Add subject labels with better positioning to avoid overlap with metrics
        for idx, row in df.iterrows():
            # Calculate position to avoid bottom-right corner where metrics are
            x_pos = ideal_allocation.iloc[idx]
            y_pos = row['Allocated_Hours']

            # Adjust label position based on location relative to plot area
            if x_pos > 0.7 * max_val and y_pos < 0.3 * max_val:
                # If in bottom-right area, move label up and left
                xytext = (-15, 15)
            else:
                xytext = (8, 8)

            ax1.annotate(row['Subject'],
                         (x_pos, y_pos),
                         xytext=xytext, textcoords='offset points',
                         fontsize=9, fontweight='bold', alpha=0.9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        ax1.set_xlabel('Ideal Allocation Based on Weakness Score (Hours)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('AI Allocation (Hours)', fontweight='bold', fontsize=12)
        ax1.set_title('AI vs Ideal Time Allocation', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Add colorbar for weakness scores
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Weakness Score', fontweight='bold', fontsize=11)

        # Calculate and display efficiency metrics
        mae = np.mean(np.abs(ideal_allocation - df['Allocated_Hours']))  # Mean Absolute Error
        rmse = np.sqrt(np.mean((ideal_allocation - df['Allocated_Hours']) ** 2))  # Root Mean Square Error
        correlation = np.corrcoef(ideal_allocation, df['Allocated_Hours'])[0, 1]
        r_squared = correlation ** 2

        # Display metrics on the plot - positioned to avoid overlap
        metrics_text = f"""Performance Metrics:
• Correlation: r = {correlation:.3f}
• R² = {r_squared:.3f}
• MAE = {mae:.2f} hours
• RMSE = {rmse:.2f} hours"""

        ax1.text(0.95, 0.05, metrics_text,
                 transform=ax1.transAxes,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.9),
                 fontsize=9, fontweight='bold',
                 verticalalignment='bottom', horizontalalignment='right')

        # Summary Statistics and Analysis
        ax2.axis('off')

        # Calculate comprehensive metrics
        total_subjects = len(df)
        weak_subjects = len(df[df['Weakness_Score'] >= 7])
        moderate_subjects = len(df[(df['Weakness_Score'] >= 4) & (df['Weakness_Score'] < 7)])
        strong_subjects = len(df[df['Weakness_Score'] < 4])

        avg_weak_time = df[df['Weakness_Score'] >= 7]['Allocated_Hours'].mean() if weak_subjects > 0 else 0
        avg_moderate_time = df[(df['Weakness_Score'] >= 4) & (df['Weakness_Score'] < 7)][
            'Allocated_Hours'].mean() if moderate_subjects > 0 else 0
        avg_strong_time = df[df['Weakness_Score'] < 4]['Allocated_Hours'].mean() if strong_subjects > 0 else 0

        time_ratio_weak_strong = avg_weak_time / avg_strong_time if avg_strong_time > 0 else float('inf')

        # Calculate allocation efficiency differently - how close AI is to ideal
        efficiency_scores = []
        for i in range(len(df)):
            if ideal_allocation.iloc[i] > 0:
                efficiency = 1 - abs(ideal_allocation.iloc[i] - df['Allocated_Hours'].iloc[i]) / ideal_allocation.iloc[
                    i]
                efficiency_scores.append(max(0, efficiency))  # Ensure non-negative

        overall_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0

        # Determine AI performance assessment
        if correlation > 0.8 and overall_efficiency > 0.8:
            performance_assessment = "🟢 EXCELLENT - AI shows strong alignment with weakness-based allocation"
        elif correlation > 0.6 and overall_efficiency > 0.6:
            performance_assessment = "🟡 GOOD - AI demonstrates reasonable weakness prioritization"
        elif correlation > 0.3:
            performance_assessment = "🟠 MODERATE - AI shows some weakness awareness but needs improvement"
        else:
            performance_assessment = "🔴 POOR - AI allocation doesn't align well with weakness scores"

        # Create detailed summary text
        summary_text = f"""
📊 COMPREHENSIVE EVALUATION SUMMARY

🎯 Study Plan Overview:
• Total Subjects: {total_subjects}
• Daily Study Time: {self.planner.total_daily_hours:.1f} hours
• Average time per subject: {self.planner.total_daily_hours / total_subjects:.1f} hours

📈 Subject Distribution by Weakness:
• Weak Subjects (≥7): {weak_subjects} subjects
• Moderate Subjects (4-6): {moderate_subjects} subjects  
• Strong Subjects (<4): {strong_subjects} subjects

⏰ AI Time Allocation Analysis:
• Avg. time for weak subjects: {avg_weak_time:.1f}h
• Avg. time for moderate subjects: {avg_moderate_time:.1f}h
• Avg. time for strong subjects: {avg_strong_time:.1f}h
• Weak:Strong ratio: {time_ratio_weak_strong:.1f}:1

🤖 AI Performance Metrics:
• Ideal-AI Correlation: r = {correlation:.3f}
• R-squared: {r_squared:.3f}
• Allocation Efficiency: {overall_efficiency:.1%}
• Mean Absolute Error: {mae:.2f} hours

{performance_assessment}

💡 Insights:
• AI {'successfully prioritizes' if correlation > 0.5 else 'struggles to prioritize'} weak subjects
• Time allocation is {'well-balanced' if 0.5 <= overall_efficiency <= 1.0 else 'needs adjustment'}
• {'Strong positive correlation indicates effective weakness-based planning' if correlation > 0.7 else 'Consider reviewing AI algorithm for better weakness detection'}
        """

        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
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
            'r_squared': r_squared,
            'efficiency': overall_efficiency,
            'mae': mae,
            'rmse': rmse,
            'time_ratio': time_ratio_weak_strong,
            'weak_subjects': weak_subjects,
            'moderate_subjects': moderate_subjects,
            'strong_subjects': strong_subjects,
            'ideal_allocation': ideal_allocation,
            'dataframe': df,
            'performance_assessment': performance_assessment
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
    planner.add_subject("Math", weakness_score=9, difficulty_pref=0.6)
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
    print(f"• R-squared: {results['r_squared']:.3f}")
    print(f"• Allocation Efficiency: {results['efficiency']:.1%}")
    print(f"• Mean Absolute Error: {results['mae']:.2f} hours")
    print(f"• Weak:Strong Time Ratio: {results['time_ratio']:.1f}:1")
    print(f"• Assessment: {results['performance_assessment']}")

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