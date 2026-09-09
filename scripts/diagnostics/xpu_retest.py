import torch, time
print("torch:", torch.__version__, "| xpu:", torch.xpu.is_available())
if not torch.xpu.is_available():
    raise SystemExit("no xpu")
print("device:", torch.xpu.get_device_name(0))
p = torch.xpu.get_device_properties(0)
print("driver:", p.driver_version, "| fp64:", p.has_fp64)
ok = 0
try:
    for i in range(150):
        n = 512 + (i % 8) * 512
        a = torch.randn(n, 128, device="xpu"); b = torch.randn(2048, 128, device="xpu")
        d = torch.cdist(a, b)
        v, idx = torch.topk(-d, 10, dim=1)
        _ = v.sum().item()
        torch.xpu.synchronize()
        ok += 1
    print("topk stress: %d/150 OK  -- the 2.13.0+xpu DEVICE_LOST crash is GONE" % ok)
except Exception as e:
    print("topk still FAILS at iter %d: %s: %s" % (ok, type(e).__name__, str(e)[:150]))
    raise SystemExit(0)
# is it now faster than CPU at this project's case-base sizes?
for n in (500, 2000):
    a = torch.randn(256,128); b = torch.randn(n,128)
    t0=time.time()
    for _ in range(200): torch.topk(-torch.cdist(a,b),10,dim=1)
    tc=time.time()-t0
    ax=a.to("xpu"); bx=b.to("xpu"); torch.xpu.synchronize()
    torch.topk(-torch.cdist(ax,bx),10,dim=1); torch.xpu.synchronize()
    t0=time.time()
    for _ in range(200): torch.topk(-torch.cdist(ax,bx),10,dim=1)
    torch.xpu.synchronize(); tx=time.time()-t0
    print("case-base n=%5d: cpu=%7.1fms xpu=%7.1fms  speedup=%.2fx" % (n, tc*1000, tx*1000, tc/tx))
