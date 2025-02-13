import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# === Constants ===
Syearly = 2         # No. of semesters per year
Wsemester = 16      # Weeks per semester
Cclass = 5          # Classes per student per semester

# Updated Student projections (Q2 and Q4 Labels)
student_projection = {
    "Q2-2025": 500, "Q4-2025": 10000,
    "Q2-2026": 3000, "Q4-2026": 6000,
    "Q2-2027": 10000, "Q4-2027": 20000,
    "Q2-2028": 40000, "Q4-2028": 70000,
    "Q2-2029": 100000, "Q4-2029": 150000
}

years = list(student_projection.keys())
students_projection = list(student_projection.values())

# Sidebar for cost parameters
st.sidebar.header("University Related Parameters (TA vs AI)")
with st.sidebar.expander("Adjust Drivers"):
    # === Sidebar Inputs ===
    # TA parameters
    ta_student_ratio    = st.slider("Student-TA Ratio", 20, 40, 30)
    Csubmission             = st.slider("Assignmet Submissions per Class", 4, 10, 5)
    default_gh = int(Csubmission*(8+ta_student_ratio*20/60)/15)
    HTA                 = st.slider("TA Grading-Hours per Week", 5, 15, default_gh)
    WTA                 = st.slider("TA Wages per Hour ($)", 9, 15, 10)
    # AI Subscription Cost
    Cai                 = st.slider("AI Subscription Cost per Student per Yr ($)", 5, 30, 10)

# Sidebar for revenue parameters
st.sidebar.header("GradeAnt (Company) Parameters")
with st.sidebar.expander("Adjust Drivers"):
    llm_cost_per_100_submissions  = st.slider("LLM Cost per 100 submissions [max 15 pages] ($)", 0.5, 5.0, 2.5)
    # Hosting and Staff Cost Inputs
    staff_cost              = st.slider("Annual Staff Cost (cummulative) ($)", 25000, 150000, 100000)
    hosting_cost_per_10k    = st.slider("Hosting Cluster Cost per 10K Students ($)", 1000, 2000, 1400)

# === X-Axis Max Value Selection ===
max_year_label  = st.sidebar.selectbox("Select Maximum X-Axis Value", years, index=len(years)-1)
max_year_index = years.index(max_year_label) + 1

# === Cost Calculations ===
def calc_univ_ta_cost(S):
    """Calculate TA grading cost for a university."""
    T_total = (S * Cclass) / ta_student_ratio
    cost = T_total * HTA * WTA * Wsemester * Syearly
    return max(cost / 1e6, 1e-6)

def calc_univ_ai_cost(S):
    """Calculate AI grading cost for a university."""
    cost = S * Cai #* Syearly
    return max(cost / 1e6, 1e-6)

def calc_grad_ai_cost(S):
    """Calculate AI-based annual grading cost for GradeAnt (including hosting)."""
    grading_cost = S * Csubmission * Syearly/ 100 * llm_cost_per_100_submissions
    hosting_cost = (S / 10000) * hosting_cost_per_10k
    total_cost = grading_cost + hosting_cost + staff_cost
    return max(total_cost / 1e6, 1e-6)

def calc_grad_revenue(S):
    """Estimate revenue for GradeAnt based on AI grading subscription model."""
    revenue = S * Cai #* Syearly * 1.5  # based on yealy subscription
    return max(revenue / 1e6, 1e-6)

# === University Cost Plot ===
def plot_university_cost_comparison(years, students_projection, scale_option, chart_type):
    ta_costs = [calc_univ_ta_cost(S) for S in students_projection]
    ai_costs = [calc_univ_ai_cost(S) for S in students_projection]
    
    fig, ax = plt.subplots()
    x = np.arange(len(years))
    
    if chart_type == "Bar":
        width = 0.35
        ax.bar(x - width/2, ta_costs, width, label="TA Grading Cost", color="brown")
        ax.bar(x + width/2, ai_costs, width, label="AI Grading Cost", color="blue")
    else:
        ax.plot(x, ta_costs, marker="o", linestyle="-", label="TA Grading Cost", color="brown")
        ax.plot(x, ai_costs, marker="o", linestyle="-", label="AI Grading Cost", color="blue")
    
    ax.set_xticks(x)
    yrlabel = [ f"{year}\n({str(student_projection[year])})" for year in years]
    ax.set_xticklabels(yrlabel, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("QtrEnd-Year (students)")
    ax.set_ylabel("Cost (Million $)")
    ax.set_title("University Grading Cost Comparison: TA-grading vs AI-grading")
    
    if scale_option == "Log-Scale":
        ax.set_yscale("log")
    
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    st.pyplot(fig)

# === GradeAnt Cost vs Revenue Plot ===
def plot_grad_cost_vs_revenue(years, students_projection, scale_option, chart_type):
    grad_ai_costs = [calc_grad_ai_cost(S) for S in students_projection]
    grad_revenues = [calc_grad_revenue(S) for S in students_projection]

    fig, ax = plt.subplots()
    x = np.arange(len(years))
    
    if chart_type == "Bar":
        width = 0.35
        ax.bar(x - width/2, grad_ai_costs, width, label="GradeAnt Cost", color="blue")
        ax.bar(x + width/2, grad_revenues, width, label="GradeAnt Revenue", color="green")
    else:
        ax.plot(x, grad_ai_costs, marker="o", linestyle="-", label="GradeAnt Cost", color="blue")
        ax.plot(x, grad_revenues, marker="o", linestyle="-", label="GradeAnt Revenue", color="green")
    
    ax.set_xticks(x)
    yrlabel = [ f"{year}\n({str(student_projection[year])})" for year in years]
    ax.set_xticklabels(yrlabel, rotation=45, ha="right", fontsize=8)
    #ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_xlabel("QtrEnd-Year (Students)")
    ax.set_ylabel("Cost / Revenue (Million $)")
    ax.set_title("GradeAnt Cost vs Revenue")
    
    if scale_option == "Log-Scale":
        ax.set_yscale("log")
    elif scale_option == "Linear":
        ax.ticklabel_format(style='plain', axis='y')
    
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    
    st.pyplot(fig)


st.subheader("University Cost Comparison: TA vs AI")
col1, col2 = st.columns([1, 1])
with col1:
    univ_scale_option = st.radio(
        "University Cost Scale (Y-Axis Only):",
        ["Linear", "Log-Scale"],
        horizontal=True
    )
with col2:
    univ_chart_type = st.radio(
        "University Chart Type:",
        ["Bar", "Line"],
        horizontal=True
    )

plot_university_cost_comparison(years[:max_year_index], students_projection[:max_year_index], univ_scale_option, univ_chart_type)

# Display University Cost Table
univ_cost_data = {
    "Quarter": years[:max_year_index],
    "Students": students_projection[:max_year_index],
    "TA Cost (Million $)": [calc_univ_ta_cost(S) for S in students_projection[:max_year_index]],
    "AI Cost (Million $)": [calc_univ_ai_cost(S) for S in students_projection[:max_year_index]]
}
st.table(univ_cost_data)

st.subheader("GradeAnt Cost vs Revenue")
col3, col4 = st.columns([1, 1])
with col3:
    grad_scale_option = st.radio(
        "GradeAnt Cost Scale (Y-Axis Only):",
        ["Linear", "Log-Scale"],
        horizontal=True
    )
with col4:
    grad_chart_type = st.radio(
        "GradeAnt Chart Type:",
        ["Bar", "Line"],
        horizontal=True
    )

plot_grad_cost_vs_revenue(years[:max_year_index], students_projection[:max_year_index], grad_scale_option, grad_chart_type)

# Display GradeAnt Cost vs Revenue Table
grad_cost_data = {
    "Quarter": years[:max_year_index],
    "Students": students_projection[:max_year_index],
    "GradeAnt Cost (Million $)": [calc_grad_ai_cost(S) for S in students_projection[:max_year_index]],
    "Revenue (Million $)": [calc_grad_revenue(S) for S in students_projection[:max_year_index]]
}
st.table(grad_cost_data)

st.title("TA vs AI Grading Cost:")
# Total TAs Calculation
st.latex(r"""
\text{Total TAs} = \frac{\text{Students} \times \text{Classes per Semester}}{\text{Student-TA Ratio}}
""")
# TA Grading Cost Calculation
st.latex(r"""
\text{TA Grading Cost} = \text{Total TAs} \times \text{TA Hours per Week} \times \text{TA Wage per Hour} 
""")
st.latex(r"""
\quad \times \text{Weeks per Semester} \times \text{Semesters per Year}
""")
# AI Grading Cost Calculation
st.latex(r"""
\text{AI Grading Cost} = \text{Students} \times \text{Annual AI Subscription Cost}
""")

st.title("GradeAnt Cost and Revenue:")
# LLM Cost Calculation
st.latex(r"""
\text{LLM Cost} = \left( \frac{\text{Students} \times \text{Assignments per Class} \times \text{Semesters per Year}}{100} \right) 
\times \text{LLM Cost per 100 Submissions}
""")
# Hosting Cost Calculation
st.latex(r"""
\text{Hosting Cost} = \left( \frac{\text{Students}}{10,000} \right) 
\times \text{Hosting Cost per 10K Students}
""")
# Total GradeAnt Cost Calculation
st.latex(r"""
\text{Total Cost} = \text{LLM Cost} + \text{Hosting Cost} + \text{Annual Staff Cost}
""")
# GradeAnt Revenue Calculation
st.latex(r"""
\text{Revenue} = \text{Students} \times \text{Annual AI Subscription Cost}
""")
