"""Serve a local reviewer engineering demo; no external service or dependencies."""
from __future__ import annotations
import argparse
import json
import os
import secrets
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NNKNN_DEVICE", "cpu")
import torch
from model.t1.reviewer_ui import ReviewerSession

HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>案例审阅工作台</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f5f0;color:#25352b;font:15px system-ui,sans-serif}main{max-width:1250px;margin:32px auto;padding:0 24px}h1{font-size:29px;margin-bottom:8px}h2{font-size:19px}p{line-height:1.7}.muted{color:#637368}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.panel{background:white;border:1px solid #d9e0d9;border-radius:12px;padding:20px}label{display:block;margin:12px 0 5px}select,input,textarea,button{font:inherit;border:1px solid #c2cec4;border-radius:6px;padding:9px;width:100%}button{background:#2c6048;color:white;cursor:pointer;margin-top:16px}button:disabled{opacity:.5}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:9px 5px;text-align:left;border-bottom:1px solid #e8ece8}th{color:#5b6b60}.shape{width:25px;height:25px;display:inline-block;vertical-align:middle;margin-right:6px}.circle{border-radius:50%}.red{background:#c4504b}.blue{background:#426eb2}.badge{background:#edf2e9;padding:4px 8px;border-radius:4px;display:inline-block}#message{white-space:pre-wrap;margin:16px 0;line-height:1.6}details{margin:14px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.wide{margin-top:20px}small{display:block;color:#637368} @media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:0 12px}table{font-size:12px}}
</style><main><h1>案例审阅工作台</h1><p class="muted">工程演示 · 红色且为圆形的物体符合条件，大小无关。已注入两处案例标签错误；每次操作都会保存真实模型和记录。</p>
<div class="grid"><section class="panel"><h2>先检查一个判断</h2><label>目标物体</label><select id="query"></select><div id="decision"></div><h2>本次实际参考的案例</h2><div id="evidence"></div><small>贡献是当前判断使用的权重；它不能单独证明修改一定有益。</small></section>
<section class="panel"><h2>记录并执行修改</h2><label>审阅者</label><input id="actor" value="本地工程审阅"><label>案例编号</label><select id="case"></select><label>操作</label><select id="operation">
<option value="relabel">更正标签</option><option value="adjust_bias">调整案例偏置</option><option value="quarantine">隔离</option><option value="release">解除隔离</option><option value="protect">保护</option><option value="unprotect">取消保护</option><option value="archive_remove">移入档案</option><option value="restore">恢复档案案例</option><option value="force">仅本次强制参考</option><option value="set_feature_weights">编辑特征参数</option><option value="set_case_weights">编辑案例参数</option><option value="undo">撤销一条参数修改</option><option value="retrain">修改后继续训练并比较</option></select>
<div id="parameters"></div><label>理由</label><textarea id="reason" rows="2" placeholder="说明为什么修改，以及期望影响哪个判断"></textarea><label>对目标判断的预期效果（操作前记录）</label><select id="expect"><option value="improve">改善</option><option value="unchanged">保持不变</option><option value="uncertain">不确定</option></select><label>判断信心 0–100</label><input id="confidence" type="number" min="0" max="100" value="80"><button id="apply">执行并保存</button><div id="message" role="status" aria-live="polite"></div></section></div>
<section class="panel wide"><h2>案例库与结果</h2><div id="cases"></div><details><summary>评分含义</summary><p>Q：本次演示判断的正确／错误贡献经平滑后的比例；B：同标签案例的偏置排名。R 是参考次数，A 是总贡献，C／H 分别是正确／错误判断中的贡献。这里的统计来自八个演示判断，不能代表广泛可靠性。保护和隔离需要审阅者明确操作。</p></details><div id="history"></div><details><summary>完整当前记录</summary><pre id="raw"></pre></details><p class="muted">参与者研究尚未开展。继续训练只执行显式的固定轮数，并另存未修改基线的同轮数对照；压缩后的运算成本可能不同。</p></section></main>
<script>
const token='__TOKEN__';let state=null;const $=id=>document.getElementById(id);const label=n=>n?'符合':'不符合';
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const object=c=>`<span class="shape ${c.color==='红色'?'red':'blue'} ${c.shape==='圆形'?'circle':''}"></span>${escape(c.color+c.shape)}`;
function parameters(){let op=$('operation').value;let h={relabel:'<label>新标签</label><select id="value"><option value="0">不符合</option><option value="1">符合</option></select>',adjust_bias:'<label>偏置改变量</label><input id="value" type="number" step="0.1" value="0.5">',force:'<label>每个强制案例的最低贡献</label><input id="value" type="number" step="0.05" min="0.05" max="1" value="0.2">',set_feature_weights:'<label>原始特征参数（颜色、形状、大小）</label><input id="value" value="[[1,1,1]]">',set_case_weights:'<label>原始案例参数</label><input id="value" value="[1]">',undo:'<label>要撤销的修改</label><select id="value">'+(state?.interventions||[]).filter(r=>['relabel','adjust_bias','set_feature_weights','set_case_weights'].includes(r.operation)).map(r=>`<option value="${escape(r.intervention_id)}">${escape(r.operation)} · ${escape(r.intervention_id)}</option>`).join('')+'</select>',retrain:'<label>固定训练轮数 1–20</label><input id="value" type="number" min="1" max="20" value="3">'}[op]||'';$('parameters').innerHTML=h;}
function decision(){let q=state.queries[Number($('query').value)];$('decision').innerHTML=`<p>${object(q)} <span class="badge">当前判断：${label(q.prediction)}</span></p><small>事件 ${escape(q.retrieval_event_id)}</small>`;$('evidence').innerHTML='<table><tr><th>编号</th><th>内容／标签</th><th>贡献</th><th>距离</th></tr>'+q.case_ids.map((id,j)=>{let c=state.cases.find(c=>c.id===id);return `<tr><td>${id}</td><td>${object(c)} · ${label(c.label)}</td><td>${q.activations[j].toFixed(3)}</td><td>${q.distances[j].toFixed(3)}</td></tr>`}).join('')+'</table>';}
function render(s){state=s;let oldq=$('query').value,oldc=$('case').value;$('query').innerHTML=s.queries.map(q=>`<option value="${q.id}">物体 ${q.id} · ${escape(q.color+q.shape)} · 大小 ${q.size.toFixed(2)}</option>`).join('');if(oldq)$('query').value=oldq;$('case').innerHTML=s.cases.map(c=>`<option value="${c.id}">案例 ${c.id} · ${escape(c.color+c.shape)}</option>`).join('')+s.archive_ids.filter(id=>!s.cases.some(c=>c.id===id)).map(id=>`<option value="${id}">档案案例 ${id}</option>`).join('');if([...$('case').options].some(o=>o.value===oldc))$('case').value=oldc;
$('cases').innerHTML='<table><tr><th>编号</th><th>物体</th><th>标签／状态</th><th>Q／B</th><th>R／A</th><th>C／H</th></tr>'+s.cases.map(c=>`<tr><td>${c.id}</td><td>${object(c)}</td><td>${label(c.label)} ${c.quarantined?'· 隔离':''} ${c.protected?'· 保护':''}</td><td>${c.Q.toFixed(2)} / ${c.B.toFixed(2)}</td><td>${c.R} / ${c.A.toFixed(2)}</td><td>${c.C.toFixed(2)} / ${c.H.toFixed(2)}</td></tr>`).join('')+'</table>';
$('history').innerHTML='<h2>操作记录</h2>'+s.actions.map(a=>`<p>v${a.version} · ${escape(a.operation)} · ${escape(a.reason)}<br><small>改正 ${a.flips.wrong_to_correct} 个判断；变错 ${a.flips.correct_to_wrong} 个判断；其余区域变错 ${a.collateral_flips.correct_to_wrong} 个判断。</small></p>`).join('');$('raw').textContent=JSON.stringify(s,null,2);$('apply').disabled=!!s.retraining;decision();parameters();}
$('query').onchange=decision;$('operation').onchange=parameters;$('apply').onclick=async()=>{let op=$('operation').value,r={version:state.version,operation:op,case_id:Number($('case').value),query_id:Number($('query').value),actor:$('actor').value,reason:$('reason').value,expected_effect:$('expect').value,confidence:Number($('confidence').value)};let v=$('value')?.value;
try{if(op==='relabel')r.label=Number(v);if(op==='adjust_bias')r.delta=Number(v);if(op==='force'){r.case_ids=[r.case_id];r.activation_floor=Number(v);}if(op.includes('weights'))r.weights=JSON.parse(v);if(op==='undo')r.intervention_id=v;if(op==='retrain')r.epochs=Number(v);$('apply').disabled=true;let res=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json','X-Reviewer-Token':token},body:JSON.stringify(r)});let s=await res.json();if(!res.ok)throw Error(s.error);render(s);$('message').textContent=op==='force'?`已记录本次强制参考，临时判断：${label(s.controlled_result.prediction)}。案例库未持久改变；完整临时事件已保存。`:`已保存 v${s.version}，可在下方检查目标及其他判断的变化。`;}catch(e){$('message').textContent=e.message;}finally{$('apply').disabled=!!state?.retraining;}};
fetch('/api/state').then(r=>r.json()).then(render).catch(e=>$('message').textContent=e.message);
</script></html>"""

# Associate visible labels with native controls, including dynamic parameters.
for _control, _label in {"query": "目标物体", "actor": "审阅者", "case": "案例编号", "operation": "操作",
                        "reason": "理由", "expect": "对目标判断的预期效果（操作前记录）",
                        "confidence": "判断信心 0–100"}.items():
    HTML = HTML.replace(f"<label>{_label}</label>", f'<label for="{_control}">{_label}</label>')
for _label in ("新标签", "偏置改变量", "每个强制案例的最低贡献", "原始特征参数（颜色、形状、大小）",
               "原始案例参数", "要撤销的修改", "固定训练轮数 1–20"):
    HTML = HTML.replace(f"<label>{_label}</label>", f'<label for="value">{_label}</label>')


def make_handler(session, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, content_type="application/json; charset=utf-8"):
            raw = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def valid_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.valid_host():
                return self.send(403, '{"error":"invalid host"}')
            if self.path == "/":
                return self.send(200, HTML.replace("__TOKEN__", token), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self.send(200, json.dumps(session.snapshot, ensure_ascii=False))
            self.send(404, '{"error":"not found"}')

        def do_POST(self):
            origin = self.headers.get("Origin")
            expected = f"http://127.0.0.1:{self.server.server_port}"
            if not self.valid_host() or self.headers.get("X-Reviewer-Token") != token or origin not in (None, expected):
                return self.send(403, '{"error":"invalid local session"}')
            if self.path != "/api/action":
                return self.send(404, '{"error":"not found"}')
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 65536:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                result = session.act(request)
                self.send(200, json.dumps(result, ensure_ascii=False))
            except (ValueError, KeyError, IndexError, StopIteration, TypeError) as exc:
                self.send(400, json.dumps({"error": str(exc)}, ensure_ascii=False))
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    torch.set_num_threads(1)
    session = ReviewerSession(args.out, seed=args.seed)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(session, secrets.token_hex(24)))
    ready = {"url": f"http://127.0.0.1:{server.server_port}", "pid": os.getpid(), "output": str(args.out.resolve())}
    (args.out / "server.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
    print(json.dumps(ready), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
