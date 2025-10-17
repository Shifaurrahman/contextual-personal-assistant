# streamlit_ui.py - Updated to match backend API responses
import streamlit as st
import requests
from typing import Dict, List, Any

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Contextual Personal Assistant", layout="wide", page_icon="🧠")

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Contextual Personal Assistant</div>', unsafe_allow_html=True)
st.markdown("*Powered by LangChain & GPT-4o-mini*")

# Sidebar with system info
with st.sidebar:
    st.header("ℹ️ System Info")
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        if health_response.status_code == 200:
            health = health_response.json()
            st.success(f"**Status:** {health['status'].title()}")
            
            if health.get('agents'):
                st.write("**Agents:**")
                st.write(f"• Ingestion: {health['agents']['ingestion']}")
                st.write(f"• Thinking: {health['agents']['thinking']}")
            
            if health.get('openai_api_key') == 'missing':
                st.error("⚠️ OpenAI API key not configured")
    except:
        st.warning("Cannot connect to backend")
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.rerun()

# ---------------- Add a new note ----------------
st.header("📝 Add a New Note")
st.caption("Enter your thoughts, tasks, or ideas - the AI will organize them automatically")

note_input = st.text_area(
    "Your Note:",
    placeholder="Examples:\n• Call Sarah about the Q3 budget next Monday\n• Idea: new logo should be blue and green\n• Remember to pick up milk on the way home",
    height=100
)

if st.button("✨ Process Note", type="primary", use_container_width=True):
    if not note_input.strip():
        st.warning("Please enter a note first.")
    else:
        with st.spinner("🤖 Processing with AI..."):
            try:
                response = requests.post(f"{BASE_URL}/add", json={"note": note_input})
                if response.status_code == 200:
                    res = response.json()
                    st.success("✅ Note processed successfully!")
                    
                    # Show key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Card Type", res.get("card_type", "Unknown"))
                    with col2:
                        st.metric("Card ID", res.get("card_id", "N/A"))
                    with col3:
                        st.metric("Envelope ID", res.get("envelope_id", "N/A"))
                    with col4:
                        new_env = "✓ New" if res.get("created_new_envelope") else "Existing"
                        st.metric("Envelope", new_env)
                    
                    # Show confidence and extracted details
                    if res.get("envelope_score"):
                        confidence_pct = int(res["envelope_score"] * 100)
                        st.progress(res["envelope_score"], text=f"Envelope Confidence: {confidence_pct}%")
                    
                    # Show extracted entities
                    extracted_info = []
                    if res.get("extracted_date"):
                        extracted_info.append(f"📅 Date: {res['extracted_date']}")
                    if res.get("extracted_time"):
                        extracted_info.append(f"⏰ Time: {res['extracted_time']}")
                    
                    if extracted_info:
                        st.info(" | ".join(extracted_info))
                    
                    # Clear the input
                    note_input = ""
                    
                elif response.status_code == 503:
                    st.error("⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY in .env file")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure the server is running:\n```uvicorn app:app --reload```")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

st.markdown("---")

# ---------------- Tabs for different views ----------------
tab1, tab2, tab3, tab4 = st.tabs(["📦 Envelopes", "📝 Cards", "👤 User Context", "🤔 Thinking Agent"])

with tab1:
    st.header("📦 Envelopes")
    st.caption("High-level topics and projects that organize your cards")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        refresh_envelopes = st.button("🔄 Load Envelopes", key="load_env")
    
    if refresh_envelopes or 'show_envelopes' in st.session_state:
        st.session_state.show_envelopes = True
        
        with st.spinner("Fetching envelopes..."):
            try:
                response = requests.get(f"{BASE_URL}/envelopes")
                if response.status_code == 200:
                    envelopes = response.json()
                    
                    if not envelopes:
                        st.info("📭 No envelopes yet. Add some notes to get started!")
                    else:
                        st.success(f"Found {len(envelopes)} envelope(s)")
                        
                        for env in envelopes:
                            card_count = env.get('card_count', 0)
                            
                            with st.expander(f"📦 **{env['name']}** (ID: {env['id']}) - {card_count} card(s)", expanded=card_count > 0):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    keywords = env.get("topic_keywords", [])
                                    if keywords:
                                        st.write("**🏷️ Keywords:**")
                                        st.write(", ".join([f"`{k}`" for k in keywords]))
                                    else:
                                        st.write("*No keywords*")
                                    
                                    st.write(f"**📅 Created:** {env.get('created_at', 'Unknown')[:10]}")
                                
                                with col2:
                                    if st.button("🗑️ Delete", key=f"del_env_{env['id']}"):
                                        if st.session_state.get(f"confirm_del_env_{env['id']}", False):
                                            # Actually delete
                                            del_resp = requests.delete(f"{BASE_URL}/envelopes/{env['id']}")
                                            if del_resp.status_code == 200:
                                                st.success("Deleted!")
                                                st.rerun()
                                            else:
                                                st.error("Delete failed")
                                        else:
                                            st.session_state[f"confirm_del_env_{env['id']}"] = True
                                            st.warning("Click again to confirm")
                else:
                    st.error("Failed to fetch envelopes")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab2:
    st.header("📝 Cards")
    st.caption("All your tasks, reminders, and ideas")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        refresh_cards = st.button("🔄 Load Cards", key="load_cards")
    with col2:
        filter_envelope = st.selectbox(
            "Filter by Envelope:",
            ["All"] + [f"Envelope {i}" for i in range(1, 20)],
            key="filter_env"
        )
    
    if refresh_cards or 'show_cards' in st.session_state:
        st.session_state.show_cards = True
        
        with st.spinner("Fetching cards..."):
            try:
                response = requests.get(f"{BASE_URL}/cards")
                if response.status_code == 200:
                    cards = response.json()
                    
                    if not cards:
                        st.info("📭 No cards yet. Add your first note above!")
                    else:
                        # Apply filter
                        if filter_envelope != "All":
                            env_id = int(filter_envelope.split()[-1])
                            cards = [c for c in cards if c.get('envelope_id') == env_id]
                        
                        st.success(f"Found {len(cards)} card(s)")
                        
                        # Group by type
                        card_types = {"Task": [], "Reminder": [], "Idea/Note": []}
                        for card in cards:
                            card_type = card.get('type', 'Idea/Note')
                            if card_type in card_types:
                                card_types[card_type].append(card)
                        
                        for card_type, cards_list in card_types.items():
                            if cards_list:
                                emoji = "✅" if card_type == "Task" else ("⏰" if card_type == "Reminder" else "💡")
                                st.subheader(f"{emoji} {card_type}s ({len(cards_list)})")
                                
                                for card in cards_list:
                                    desc_preview = card['description'][:80] + "..." if len(card['description']) > 80 else card['description']
                                    
                                    with st.expander(f"**Card #{card['id']}:** {desc_preview}"):
                                        st.write(f"**📝 Description:** {card['description']}")
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"**📦 Envelope ID:** {card.get('envelope_id', 'None')}")
                                            st.write(f"**👤 Assignee:** {card.get('assignee') or '*Not specified*'}")
                                        with col2:
                                            st.write(f"**📅 Date:** {card.get('date') or '*Not specified*'}")
                                            st.write(f"**⏰ Time:** {card.get('time') or '*Not specified*'}")
                                        
                                        keywords = card.get('context_keywords', [])
                                        if keywords:
                                            st.write("**🏷️ Keywords:**", ", ".join([f"`{k}`" for k in keywords]))
                                        
                                        st.caption(f"Created: {card.get('created_at', 'Unknown')}")
                                        
                                        if st.button("🗑️ Delete Card", key=f"del_card_{card['id']}"):
                                            del_resp = requests.delete(f"{BASE_URL}/cards/{card['id']}")
                                            if del_resp.status_code == 200:
                                                st.success("Deleted!")
                                                st.rerun()
                else:
                    st.error("Failed to fetch cards")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab3:
    st.header("👤 User Context")
    st.caption("Your dynamic profile built from all your notes")
    
    if st.button("🔄 Refresh Context", key="load_context"):
        with st.spinner("Fetching user context..."):
            try:
                response = requests.get(f"{BASE_URL}/context")
                if response.status_code == 200:
                    context = response.json()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Active Projects")
                        projects = context.get("active_projects", [])
                        if projects:
                            for p in projects:
                                st.write(f"• **{p}**")
                        else:
                            st.info("No active projects tracked yet")
                        
                        st.markdown("### 👥 Contacts")
                        contacts = context.get("contacts", [])
                        if contacts:
                            for c in contacts:
                                st.write(f"• **{c}**")
                        else:
                            st.info("No contacts tracked yet")
                    
                    with col2:
                        st.markdown("### 📅 Upcoming Deadlines")
                        deadlines = context.get("upcoming_deadlines", [])
                        if deadlines:
                            for d in deadlines:
                                st.write(f"• {d}")
                        else:
                            st.info("No deadlines tracked yet")
                        
                        st.markdown("### 🏷️ Themes")
                        themes = context.get("themes", [])
                        if themes:
                            # Display as tags
                            theme_tags = " ".join([f"`{t}`" for t in themes[:15]])
                            st.markdown(theme_tags)
                        else:
                            st.info("No themes identified yet")
                    
                    # Show raw JSON in expander
                    with st.expander("🔍 View Raw JSON"):
                        st.json(context)
                else:
                    st.error("Failed to fetch user context")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab4:
    st.header("🤔 Thinking Agent Insights")
    st.caption("AI-powered analysis of your cards to find patterns, conflicts, and opportunities")
    
    # View mode toggle
    view_mode = st.radio(
        "Display Mode:",
        ["🤖 Natural Language (AI Summary)", "📊 Structured Data"],
        horizontal=True
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        run_thinking = st.button("🧠 Run Analysis", type="primary", use_container_width=True)
    
    if run_thinking:
        with st.spinner("🧠 Analyzing all your cards... This may take a moment..."):
            try:
                response = requests.get(f"{BASE_URL}/think?natural=true")
                
                if response.status_code == 200:
                    data = response.json()
                    insights = data.get("insights", {})
                    natural_text = data.get("natural_text", "")
                    
                    # Count non-empty insights
                    non_empty_count = sum(1 for v in insights.values() if isinstance(v, list) and len(v) > 0)
                    
                    if view_mode == "🤖 Natural Language (AI Summary)":
                        st.markdown("### 📝 AI-Generated Analysis")
                        st.markdown(natural_text)
                        
                        with st.expander("🔍 View Raw Structured Data"):
                            st.json(insights)
                    
                    else:  # Structured view
                        st.markdown("### 📊 Detailed Insights")
                        
                        if non_empty_count == 0:
                            st.success("✨ No significant issues found! Your cards are well organized.")
                        else:
                            st.info(f"Found {non_empty_count} insight category(ies) with data")
                        
                        # Priority Tasks
                        if insights.get("priority_tasks"):
                            st.markdown("#### ⚡ Priority Tasks (Next 3 Days)")
                            for task in insights["priority_tasks"]:
                                urgency_icon = "🔴" if task.get("urgency") == "High" else "🟡"
                                st.markdown(f"{urgency_icon} **{task.get('description', 'Unknown task')}**")
                                
                                info_parts = []
                                if task.get('due_date'):
                                    info_parts.append(f"📅 Due: {task['due_date']}")
                                if task.get('days_until') is not None:
                                    info_parts.append(f"⏳ In {task['days_until']} day(s)")
                                if task.get('assignee'):
                                    info_parts.append(f"👤 {task['assignee']}")
                                
                                if info_parts:
                                    st.caption(" • ".join(info_parts))
                                st.write("")
                        
                        # Overdue Tasks
                        if insights.get("overdue_tasks"):
                            st.markdown("#### ⏰ Overdue Tasks")
                            for task in insights["overdue_tasks"]:
                                st.error(f"**Card #{task.get('id')}:** {task.get('description', 'Unknown')}")
                                st.caption(f"📅 Due: {task.get('due_date')} • {task.get('days_overdue', 0)} day(s) overdue")
                        
                        # Scheduling Conflicts
                        if insights.get("conflicts"):
                            st.markdown("#### ⚠️ Scheduling Conflicts")
                            for conflict in insights["conflicts"]:
                                st.warning(f"**{conflict.get('assignee', 'Unknown')}** has multiple tasks on **{conflict.get('date', 'unknown date')}**:")
                                for task in conflict.get("tasks", []):
                                    st.write(f"   • Card #{task.get('id')}: {task.get('description', 'Unknown')}")
                        
                        # Potential Duplicates
                        if insights.get("potential_duplicates"):
                            st.markdown("#### 🔄 Potential Duplicate Tasks")
                            for dup in insights["potential_duplicates"][:10]:  # Limit display
                                card_a = dup.get('card_a', {})
                                card_b = dup.get('card_b', {})
                                similarity = dup.get('similarity', 0)
                                
                                st.info(f"**Card #{card_a.get('id')}** ↔️ **Card #{card_b.get('id')}** ({int(similarity*100)}% similar)")
                                st.caption(f"• {card_a.get('description', 'N/A')[:60]}...")
                                st.caption(f"• {card_b.get('description', 'N/A')[:60]}...")
                        
                        # Merge Suggestions
                        if insights.get("merge_suggestions"):
                            st.markdown("#### 🔗 Envelope Merge Suggestions")
                            for merge in insights["merge_suggestions"]:
                                env_a = merge.get('envelope_a', {})
                                env_b = merge.get('envelope_b', {})
                                reason = merge.get('reason', 'Similar content')
                                
                                st.info(f"Consider merging **{env_a.get('name', 'Unknown')}** and **{env_b.get('name', 'Unknown')}**")
                                st.caption(f"Reason: {reason}")
                        
                        # Next Steps
                        if insights.get("next_steps"):
                            st.markdown("#### 💡 Suggested Next Steps")
                            for step in insights["next_steps"]:
                                env = step.get('envelope', {})
                                suggestion = step.get('suggestion', 'N/A')
                                st.write(f"• **{env.get('name', 'General')}:** {suggestion}")
                        
                        # Show natural summary in expander
                        with st.expander("🤖 View AI-Generated Summary"):
                            st.markdown(natural_text)
                
                elif response.status_code == 503:
                    st.error("⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY in .env file")
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    st.error(f"Failed to run thinking agent: {error_detail}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure the server is running.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
        🚀 Powered by <b>LangChain</b>, <b>OpenAI GPT-4o-mini</b>, and <b>Streamlit</b><br>
        Built for the Machine Learning Engineer Assignment
    </small>
</div>
""", unsafe_allow_html=True)