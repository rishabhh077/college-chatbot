import base64
import itertools
from pathlib import Path

import streamlit as st

import rag

ASSETS = Path(__file__).parent / "assets"


def data_uri(name: str, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()


st.set_page_config(page_title="LIET Assistant", page_icon=str(ASSETS / "liet-avatar.png"), layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp {font-family: 'Inter', system-ui, sans-serif;}
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"] {visibility: visible !important; color: #E6EAF2;}
    html, body {background: #04060B;}
    .stApp {background: transparent !important;}
    .block-container {padding-top: 1.4rem; max-width: 860px;}

    /* top bar */
    .topbar {display: flex; align-items: center; gap: 14px; padding: 14px 18px; margin-bottom: 14px;
        background: rgba(7,10,18,.66); backdrop-filter: blur(12px); border: 1px solid rgba(110,130,180,.14); border-radius: 14px; animation: fade .5s ease both;}
    .logo {width: 48px; height: 48px; border-radius: 12px; display: grid; place-items: center; flex: none; background: #fff;
        box-shadow: 0 6px 18px rgba(0,0,0,.45);}
    .logo img {width: 40px; height: 40px; object-fit: contain;}
    .wordmark {height: 34px; margin: 2px 0 6px;}
    .topbar h1 {margin: 0; font-size: 1.15rem; font-weight: 650; color: #F1F5F9; letter-spacing: -.2px;}
    .topbar p {margin: 2px 0 0; font-size: .82rem; color: #8FA0BC;}
    .status {margin-left: auto; display: flex; align-items: center; gap: 7px; font-size: .78rem; color: #A7F3D0;
        background: rgba(16,185,129,.10); border: 1px solid rgba(16,185,129,.25); padding: 5px 11px; border-radius: 999px;}
    .dot {width: 7px; height: 7px; border-radius: 50%; background: #10B981; animation: ping 2s infinite;}
    .headline {line-height: 1.25em; font-size: 1.9rem; font-weight: 700; color: #F1F5F9; letter-spacing: -.6px; line-height: 1.25; margin: 18px 2px 4px;
        animation: rise .6s ease both;}
    .rotator {display: inline-block; height: 1.25em; line-height: 1.25em; overflow: hidden; vertical-align: top;}
    .rotator ul {list-style: none !important; margin: 0 !important; padding: 0 !important; animation: roll 12s cubic-bezier(.7,0,.2,1) infinite;}
    .rotator, .rotator ul, .rotator li {font-size: 1.9rem !important; font-weight: 700 !important;}
    .rotator li {display: block; height: 1.25em; line-height: 1.25em; margin: 0 !important; padding: 0 !important; background: linear-gradient(90deg, #F58220, #FBBF24); -webkit-background-clip: text;
        background-clip: text; -webkit-text-fill-color: transparent;}
    .sub {color: #8FA0BC; font-size: .95rem; margin: 0 2px 16px; animation: rise .8s ease both;}
    .feats {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 8px;}
    .feat {position: relative; overflow: hidden; background: rgba(7,10,18,.66); backdrop-filter: blur(12px);
        border: 1px solid rgba(110,130,180,.14); border-radius: 14px; padding: 14px 15px; animation: rise .7s ease both;
        transition: transform .2s ease, border-color .2s ease;}
    .feat:nth-child(2) {animation-delay: .1s;} .feat:nth-child(3) {animation-delay: .2s;}
    .feat:hover {transform: translateY(-3px); border-color: rgba(245,130,32,.6);}
    .feat::before {content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
        background: linear-gradient(90deg, transparent, #F58220, transparent); transform: translateX(-100%);
        animation: scan 4s ease-in-out infinite;}
    .feat:nth-child(2)::before {animation-delay: 1.3s;} .feat:nth-child(3)::before {animation-delay: 2.6s;}
    .ico {width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; margin-bottom: 10px;
        background: rgba(245,130,32,.14); border: 1px solid rgba(245,130,32,.3);}
    .ico svg {width: 18px; height: 18px; stroke: #FDBA74; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;}
    .feat b {display: block; color: #F1F5F9; font-size: .92rem; font-weight: 600;}
    .feat span {display: block; color: #8FA0BC; font-size: .8rem; line-height: 1.4; margin-top: 3px;}
    @media (max-width: 640px) {.feats {grid-template-columns: 1fr;} .headline, .rotator, .rotator ul, .rotator li {font-size: 1.5rem !important;}}

    [data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {background: transparent !important;}

    /* chat */
    .stChatMessage {background: rgba(7,10,18,.72); backdrop-filter: blur(12px); border: 1px solid rgba(110,130,180,.14); border-radius: 14px; animation: rise .3s ease both;}
    [data-testid="stChatInput"] {border-radius: 14px; border: 1px solid #1B2540; transition: border-color .2s, box-shadow .2s;}
    [data-testid="stChatInput"]:focus-within {border-color: #F58220; box-shadow: 0 0 0 3px rgba(245,130,32,.18);}
    .note {text-align: center; color: #6B7C99; font-size: .74rem; margin-top: 6px;}

    /* processing animation */
    .proc {padding: 4px 2px;}
    .step {display: flex; align-items: center; gap: 10px; font-size: .88rem; color: #B6C2D9; margin: 7px 0; opacity: 0;
        animation: rise .4s ease forwards;}
    .step:nth-child(2) {animation-delay: .7s;} .step:nth-child(3) {animation-delay: 1.5s;}
    .spin {width: 14px; height: 14px; border-radius: 50%; border: 2px solid #222E4A; border-top-color: #F58220;
        animation: rot .8s linear infinite; flex: none;}
    .bar {height: 3px; border-radius: 3px; margin-top: 10px; background: #141B2B; overflow: hidden;}
    .bar::after {content: ""; display: block; height: 100%; width: 40%; border-radius: 3px;
        background: linear-gradient(90deg, transparent, #F58220, transparent); animation: sweep 1.3s ease-in-out infinite;}

    /* sidebar */
    [data-testid="stSidebar"] {background: rgba(5,8,14,.9); backdrop-filter: blur(14px); border-right: 1px solid #141B2B;}
    .stButton, .stLinkButton, [data-testid="stButton"], [data-testid="stLinkButton"],
    [data-testid="stElementContainer"]:has(> [data-testid="stButton"]),
    [data-testid="stElementContainer"]:has(> [data-testid="stLinkButton"]) {width: 100% !important;}
    .stButton > button, .stLinkButton > a, [data-testid="stButton"] > button, [data-testid="stLinkButton"] > a {width: 100% !important; justify-content: flex-start; text-align: left;
        border-radius: 10px; border: 1px solid #141B2B; background: #090D17; color: #D5DDEC; font-weight: 500;
        transition: all .18s ease;}
    .stButton > button:hover, .stLinkButton > a:hover, [data-testid="stButton"] > button:hover, [data-testid="stLinkButton"] > a:hover {
        border-color: #F58220; color: #FDBA74; background: #0D1424; transform: translateX(2px);}
    .label {font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; color: #6B7C99; margin: 18px 0 6px; font-weight: 600;}
    .brand {font-size: 1.05rem; font-weight: 650; color: #F1F5F9;}
    .contact {background: #090D17; border: 1px solid #141B2B; border-radius: 10px; padding: 11px 13px;
        font-size: .82rem; line-height: 1.75; color: #B6C2D9; margin-top: 14px;}
    .follow {color: #6B7C99; font-size: .76rem; letter-spacing: .08em; text-transform: uppercase; margin: 16px 0 6px; font-weight: 600;}
    h4 {color: #F1F5F9; font-weight: 600;}

    @keyframes roll {0%,18% {transform: translateY(0);} 25%,43% {transform: translateY(-1.25em);} 50%,68% {transform: translateY(-2.5em);} 75%,93% {transform: translateY(-3.75em);} 100% {transform: translateY(-5em);}}
    @keyframes scan {0% {transform: translateX(-100%);} 45%,100% {transform: translateX(100%);}}
    @keyframes fade {from {opacity: 0;} to {opacity: 1;}}
    @keyframes rise {from {opacity: 0; transform: translateY(8px);} to {opacity: 1; transform: none;}}
    @keyframes rot {to {transform: rotate(360deg);}}
    @keyframes sweep {from {transform: translateX(-100%);} to {transform: translateX(260%);}}
    @keyframes ping {70% {box-shadow: 0 0 0 8px rgba(16,185,129,0);} 100% {box-shadow: 0 0 0 0 rgba(16,185,129,0);}}
    </style>
    """,
    unsafe_allow_html=True,
)

TOPICS = {
    "Courses & Eligibility": (":material/school:", "Which courses does LIET offer and what is the eligibility?"),
    "Admission Process": (":material/assignment:", "What is the admission process at LIET?"),
    "Fees & Scholarships": (":material/payments:", "What are the fees and scholarships at LIET?"),
    "Hostel & Facilities": (":material/apartment:", "What hostel and campus facilities does LIET provide?"),
    "Placements": (":material/work:", "What is the placement process and which companies recruit at LIET?"),
    "Library": (":material/local_library:", "What are the library timings and facilities?"),
    "Research & Innovation": (":material/science:", "What research and innovation activities happen at LIET?"),
    "Contact Details": (":material/call:", "How can I contact LIET admissions?"),
}
TOPIC_QUESTIONS = {q for _, q in TOPICS.values()}
USER_AVATAR, BOT_AVATAR = ":material/person:", str(ASSETS / "liet-avatar.png")
PROCESSING = """
<div class="proc">
  <div class="step"><div class="spin"></div>Understanding your question</div>
  <div class="step"><div class="spin"></div>Searching the LIET knowledge base</div>
  <div class="step"><div class="spin"></div>Writing your answer</div>
  <div class="bar"></div>
</div>
"""


SHADER_BG = """<script>
(function () {
  const P = window.parent, D = P.document;
  if (P.__lietBg) { P.cancelAnimationFrame(P.__lietBg.raf); P.__lietBg.canvas.remove(); }
  if (P.matchMedia && P.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const canvas = D.createElement('canvas');
  canvas.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;opacity:.8';
  D.body.prepend(canvas);
  const gl = canvas.getContext('webgl', {antialias: false, alpha: false});
  if (!gl) return;

  const vs = 'attribute vec2 p; void main(){ gl_Position = vec4(p, 0., 1.); }';
  const fs = `
    precision highp float;
    uniform vec2 res; uniform float t;
    void main() {
      vec2 uv = (gl_FragCoord.xy * 2. - res) / min(res.x, res.y);
      vec3 base = vec3(.016, .024, .043);
      vec3 col = base;
      for (int i = 0; i < 3; i++) {
        float f = float(i);
        vec2 c = vec2(cos(t * .07 + f * 2.1) * .8, sin(t * .055 + f * 1.7) * .5);
        float d = length(uv - c);
        float wave = .5 + .5 * sin(d * 3.2 - t * .22 + f * 1.3);
        float soft = pow(wave, 7.) * .9 + pow(wave, 2.) * .10;
        vec3 tint = f < .5 ? vec3(.80, .42, .16) : (f < 1.5 ? vec3(.32, .35, .80) : vec3(.12, .55, .68));
        col += tint * soft * (.34 / (.6 + d * .7));
      }
      float vig = smoothstep(2.0, .2, length(uv));
      gl_FragColor = vec4(mix(base, col, .5 + .5 * vig), 1.);
    }`;
  function sh(type, src) { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog); gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'p');
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const uRes = gl.getUniformLocation(prog, 'res'), uT = gl.getUniformLocation(prog, 't');

  const SCALE = .5;  // render at half resolution: smooth and light on the GPU
  function resize() {
    canvas.width = Math.max(2, Math.floor(P.innerWidth * SCALE));
    canvas.height = Math.max(2, Math.floor(P.innerHeight * SCALE));
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  resize(); P.addEventListener('resize', resize);

  const t0 = performance.now();
  const state = P.__lietBg = {canvas, raf: 0};
  (function frame() {
    if (!D.hidden) {
      gl.uniform2f(uRes, canvas.width, canvas.height);
      gl.uniform1f(uT, (performance.now() - t0) / 1000);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    state.raf = P.requestAnimationFrame(frame);
  })();
})();
</script>
"""

st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)


@st.cache_resource
def warm() -> bool:
    rag.warmup()
    return True


warm()
st.iframe(SHADER_BG, height=1)

with st.sidebar:
    st.markdown(
        f'<img class="wordmark" src="{data_uri("liet-logo.webp", "image/webp")}" alt="LIET">'
        '<div class="brand">LIET Assistant</div>',
        unsafe_allow_html=True,
    )
    st.caption("Lloyd Institute of Engineering & Technology")
    if st.button("New chat", icon=":material/add:", key="new"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    st.markdown('<div class="label">Quick topics</div>', unsafe_allow_html=True)
    for label, (icon, q) in TOPICS.items():
        if st.button(label, icon=icon, key=f"t_{label}"):
            st.session_state.pending = q

    st.markdown('<div class="label">Answer style</div>', unsafe_allow_html=True)
    style_label = st.radio("Answer style", [":material/bolt: Concise", ":material/menu_book: Detailed"], label_visibility="collapsed", horizontal=True)
    style = "short" if "Concise" in style_label else "detailed"

    st.markdown('<div class="label">Quick links</div>', unsafe_allow_html=True)
    st.link_button("College website", "https://liet.in", icon=":material/language:", use_container_width=True)
    st.link_button("Apply now", "https://admissions.liet.in", icon=":material/how_to_reg:", use_container_width=True)
    st.markdown(
        '<div class="contact">admissions@liet.in<br>+91 9821582662<br>Greater Noida, Delhi NCR</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="topbar">
      <div class="logo"><img src="{ICON}" alt="LIET"></div>
      <div><h1>LIET Assistant</h1><p>Admissions, courses, fees, hostel and placements, answered instantly.</p></div>
      <div class="status"><span class="dot"></span>Available</div>
    </div>
    """.replace("{ICON}", data_uri("liet-icon.png", "image/png")),
    unsafe_allow_html=True,
)
question = st.chat_input("Ask a question about LIET…") or st.session_state.pending
st.session_state.pending = None
welcome = not st.session_state.messages and not question

if not rag.index_exists():
    st.error("The knowledge base is not ready yet. Please contact the administrator.")
    st.stop()

if welcome:
    st.markdown(
        """
        <div class="headline">Ask me about
          <span class="rotator"><ul><li>Admissions</li><li>Courses</li><li>Scholarships</li><li>Placements</li><li>Admissions</li></ul></span>
        </div>
        <div class="sub">Get clear, reliable answers about life at LIET in seconds.</div>
        <div class="feats">
          <div class="feat"><div class="ico"><svg viewBox="0 0 24 24"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg></div>
            <b>Instant answers</b><span>Straight answers to your questions, with no searching through pages.</span></div>
          <div class="feat"><div class="ico"><svg viewBox="0 0 24 24"><path d="M12 3 4 6v6c0 5 3.4 8 8 9 4.6-1 8-4 8-9V6z"/><path d="m9 12 2 2 4-4"/></svg></div>
            <b>Trusted information</b><span>Built on official LIET content, not guesses from the internet.</span></div>
          <div class="feat"><div class="ico"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg></div>
            <b>Always available</b><span>Ask any time, day or night, as many questions as you like.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### How can I help you today?")
    cols = st.columns(2)
    for i, (label, (icon, q)) in enumerate(list(TOPICS.items())[:6]):
        if cols[i % 2].button(label, icon=icon, key=f"c_{label}"):
            st.session_state.pending = q
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=USER_AVATAR if m["role"] == "user" else BOT_AVATAR):
        st.markdown(m["content"])

if question:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        holder = st.empty()
        holder.markdown(PROCESSING, unsafe_allow_html=True)
        try:
            standalone = question in TOPIC_QUESTIONS or not st.session_state.messages
            pieces = rag.answer_stream(question, st.session_state.messages, style, use_cache=standalone)
            first = next(pieces)  # animation plays until the first words arrive
            holder.empty()
            reply = st.write_stream(itertools.chain([first], pieces))
        except Exception:
            holder.empty()
            reply = "Sorry, I couldn't get an answer right now. Please try again in a moment."
            st.markdown(reply)
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply},
    ]

if st.session_state.messages:
    asked = {m["content"] for m in st.session_state.messages if m["role"] == "user"}
    options = [(l, i, q) for l, (i, q) in TOPICS.items() if q not in asked][:3]
    if options:
        st.markdown('<div class="follow">Suggested next</div>', unsafe_allow_html=True)
        cols = st.columns(len(options))
        for col, (label, icon, q) in zip(cols, options):
            if col.button(label, icon=icon, key=f"f_{len(st.session_state.messages)}_{label}"):
                st.session_state.pending = q
                st.rerun()

st.markdown(
    '<div class="note">Answers are generated from LIET website content. Please confirm important details with the admissions office.</div>',
    unsafe_allow_html=True,
)
