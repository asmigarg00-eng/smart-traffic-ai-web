import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time

st.set_page_config(page_title="Smart Traffic AI", page_icon="🚦", layout="wide")
st.markdown("""
<style>
.stApp {background:#101827;color:#f4f7fb}
section[data-testid="stSidebar"] {background:#1c2940}
div[data-testid="stMetric"] {background:#263445;padding:14px;border-radius:12px}
</style>
""", unsafe_allow_html=True)
st.title("🚦 Smart Traffic AI")
st.caption("Vehicle Detection & Four-Way Signal Simulation | Educational prototype — no real signal control")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

if "queues" not in st.session_state:
    st.session_state.queues={"North":0,"East":0,"South":0,"West":0}
if "active" not in st.session_state: st.session_state.active="North"
if "green_until" not in st.session_state: st.session_state.green_until=0
if "last_counts" not in st.session_state: st.session_state.last_counts=None

with st.sidebar:
    st.header("Project menu")
    page=st.radio("Navigate",["Dashboard","How It Works","About"])
    st.divider()
    st.caption("AI model: YOLOv8n • Confidence threshold: 0.35")

if page=="How It Works":
    st.subheader("How the prototype works")
    st.markdown("""
    1. Upload a traffic image.
    2. YOLOv8 detects cars, motorcycles, buses and trucks.
    3. Review the detection count and confidence values.
    4. Manually assign that image's count to a road (direction cannot be inferred reliably from one image).
    5. The busiest assigned queue is selected for a simulated green phase; other roads are red.
    """)
    st.info("This is an image-based educational simulation, not live traffic monitoring or real signal control.")
elif page=="About":
    st.subheader("About this project")
    st.write("AI-Based Smart Traffic Management System")
    st.write("A Class XII AI project demonstrating object detection and a simple queue-based signal decision.")
    st.write("Tools: Python, Streamlit, YOLOv8, OpenCV, Pillow")
    st.warning("The system does not control physical traffic lights.")
else:
    left,right=st.columns([1.25,1])
    with left:
        st.subheader("AI Vehicle Detection")
        uploaded=st.file_uploader("Upload traffic image",type=["jpg","jpeg","png","bmp"])
        if uploaded:
            image=Image.open(uploaded).convert("RGB")
            st.image(image,caption="Uploaded traffic image",use_container_width=True)
            if st.button("🔍 Detect Vehicles",type="primary"):
                with st.spinner("Loading YOLO and analysing image…"):
                    try:
                        model=load_model()
                        arr=np.array(image)
                        result=model.predict(source=arr,conf=0.35,verbose=False)[0]
                        counts={"car":0,"motorcycle":0,"bus":0,"truck":0}
                        rows=[]
                        for box in result.boxes:
                            name=model.names[int(box.cls[0])]
                            if name in counts:
                                counts[name]+=1
                                rows.append({"Vehicle":name.title(),"Confidence":round(float(box.conf[0]),2)})
                        total=sum(counts.values())
                        level="LOW" if total<=2 else "MEDIUM" if total<=5 else "HIGH"
                        seconds=15 if total<=2 else 30 if total<=5 else 45
                        plotted=result.plot()
                        st.session_state.last_counts={"counts":counts,"total":total,"level":level,"seconds":seconds,"rows":rows,"plot":cv2.cvtColor(plotted,cv2.COLOR_BGR2RGB)}
                    except Exception as e:
                        st.error(f"Detection error: {e}")
        if st.session_state.last_counts:
            data=st.session_state.last_counts
            st.image(data["plot"],caption="YOLO detection preview",use_container_width=True)
            a,b,c,d=st.columns(4)
            a.metric("Cars",data["counts"]["car"]); b.metric("Motorcycles",data["counts"]["motorcycle"])
            c.metric("Buses",data["counts"]["bus"]); d.metric("Trucks",data["counts"]["truck"])
            a,b,c=st.columns(3); a.metric("Total",data["total"]); b.metric("Traffic level",data["level"]); c.metric("Suggested green",f'{data["seconds"]} sec')
            st.dataframe(data["rows"],use_container_width=True)
    with right:
        st.subheader("Four-Way Signal Simulation")
        roads=["North","East","South","West"]
        st.caption("Assign the detected count to the road represented by the uploaded image.")
        road=st.selectbox("Road direction",roads)
        count=st.session_state.last_counts["total"] if st.session_state.last_counts else 0
        if st.button("Apply detected count to selected road",use_container_width=True):
            if st.session_state.last_counts is None: st.warning("Detect vehicles first.")
            else: st.session_state.queues[road]=count; st.success(f"Assigned {count} to {road}.")
        if st.button("＋ Add one demo vehicle",use_container_width=True):
            st.session_state.queues[road]+=1
        if st.button("🚦 Auto-select busiest road",type="primary",use_container_width=True):
            q=st.session_state.queues
            if max(q.values())==0: st.warning("Queues are zero. Assign a count or add demo traffic.")
            else:
                m=max(q.values()); ties=[r for r in roads if q[r]==m]
                st.session_state.active=st.session_state.active if st.session_state.active in ties else ties[0]
                duration=max(15,min(60,10+m*3)); st.session_state.green_until=time.time()+duration
                st.success(f'{st.session_state.active} selected • green phase {duration}s')
        if st.button("Reset queues",use_container_width=True):
            st.session_state.queues={r:0 for r in roads}; st.session_state.active="North"; st.session_state.green_until=0
        st.markdown("### Signal status")
        active=st.session_state.active if time.time()<st.session_state.green_until else st.session_state.active
        for r in roads:
            col1,col2,col3=st.columns([1,2,1])
            col1.markdown("🟢" if r==active and st.session_state.green_until>time.time() else "🔴")
            col2.write(f"**{r}**")
            col3.write(f"Queue: {st.session_state.queues[r]}")
        st.info("Green road is chosen from the largest manually assigned queue. Other roads remain red in this simulation.")
        if st.session_state.green_until>time.time():
            st.success(f"Green phase running • {int(st.session_state.green_until-time.time())} seconds remaining")
            time.sleep(1); st.rerun()
st.divider()
st.caption("Student AI project • Prototype only • No live camera feed or physical traffic-light control")
