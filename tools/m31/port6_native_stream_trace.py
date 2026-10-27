#!/usr/bin/env python3
"""Audit native grouped-MMQ/H2D CUDA activity overlap without host-event inference."""
import argparse
import collections
import json
from pathlib import Path
import sqlite3
from port6_stream_trace import length, union


def intersections(a,b):
    a=union(a);b=union(b);i=j=0;out=[]
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
        if lo<hi:out.append((lo,hi))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--expected-grouped-gemms',type=int,default=171)
    p.add_argument('--expected-prefetch-bytes',type=int,default=160620871680)
    a=p.parse_args()
    con=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True)
    names=dict(con.execute('select id,value from StringIds'))
    streams=collections.defaultdict(list)
    for lo,hi,d,c,s,pid,name in con.execute('select start,end,deviceId,contextId,streamId,globalPid,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
        if hi>lo:streams[(d,c,pid,s)].append((lo,hi,names.get(name,'')))
    groups=[]
    for key,kernels in streams.items():
        for i,k in enumerate(kernels):
            if 'mm_ids_helper<' not in k[2]:continue
            part=[k];matched=False
            for other in kernels[i+1:i+12]:
                if 'mm_ids_helper<' in other[2]:break
                if 'mul_mat_q' in other[2]:part.append(other);matched=True
                elif matched:break
                elif 'quantize' in other[2]:part.append(other)
                else:break
            if not matched:raise RuntimeError(f'unmatched grouped-MMQ anchor: {key} {k}')
            groups.append({'key':key,'kernels':part})
    if len(groups)!=a.expected_grouped_gemms:
        raise RuntimeError(f'grouped GEMM count {len(groups)} != expected {a.expected_grouped_gemms}; refusing expert labels')
    first=min(k[0] for g in groups for k in g['kernels'])
    last=max(k[1] for g in groups for k in g['kernels'])
    enums=dict((name,idx) for idx,name in con.execute('select id,name from ENUM_CUDA_MEMCPY_OPER'))
    copies=collections.defaultdict(list);d2h=[]
    for lo,hi,d,c,s,pid,size,kind in con.execute('select start,end,deviceId,contextId,streamId,globalPid,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY'):
        if hi<=lo:continue
        if kind==enums['CUDA_MEMCPY_KIND_HTOD']:
            copies[(d,c,pid,s)].append((lo,hi,size))
        if kind==enums['CUDA_MEMCPY_KIND_DTOH'] and lo>=first and hi<=last:d2h.append((lo,hi,size))
    rows=[];all_overlap=[];expert_compute=[];examples=[]
    grouped_streams={g['key'] for g in groups}
    weight_stream_bytes=[];weight_dma=[]
    kernel_types=collections.defaultdict(list)
    for group in groups:
        for lo,hi,name in group['kernels']:
            label='helper' if 'mm_ids_helper<' in name else 'quantize' if 'quantize' in name else 'GEMM_fixup' if 'fixup' in name else 'GEMM'
            kernel_types[label].append((lo,hi))
    for g in groups:expert_compute.extend((lo,hi) for lo,hi,_ in g['kernels'])
    for key,items in sorted(copies.items()):
        large_bytes=sum(size for lo,hi,size in items if size>=1024*1024 and hi>first and lo<last)
        weight_stream=key not in grouped_streams and large_bytes>=200*1024*1024
        if weight_stream:
            weight_stream_bytes.append(sum(size for _,_,size in items))
            weight_dma.extend((max(lo,first),min(hi,last)) for lo,hi,_ in items)
        compute=[(lo,hi) for g in groups if g['key'][:3]==key[:3] and g['key'][3]!=key[3] for lo,hi,_ in g['kernels']]
        inside=[(lo,hi,size) for lo,hi,size in items if lo>=first and hi<=last]
        overlap=intersections([(max(lo,first),min(hi,last)) for lo,hi,_ in items],compute)
        if weight_stream:all_overlap.extend(overlap)
        activity=sum(hi-lo for lo,hi,_ in inside)
        rows.append(dict(weight_stream=weight_stream,whole_trace_bytes=sum(size for _,_,size in items),device=key[0],context=key[1],process=key[2],stream=key[3],
            completely_inside_grouped_window_bytes=sum(size for _,_,size in inside),copies=len(inside),
            dma_activity_ns=activity,active_GB_s=sum(size for _,_,size in inside)/activity if activity else None,
            grouped_kernel_overlap_union_ns=length(overlap)))
        if overlap:examples.append({'copy_stream':key[3],'first_overlap_ns':overlap[0],'last_overlap_ns':overlap[-1]})
    overlap=length(all_overlap);compute=length(expert_compute)
    result={'scope':'Actual native grouped-MMQ anchor/quant/GEMM kernel activities intersect actual H2D on different same-device/context/process streams; no host event inference. A kernel subset, not all FFN wall time. Weight streams must differ from reader streams and carry >=200MiB in >=1MiB copies; all their fragments count. Window excludes startup; boundary-copy bytes omitted from throughput.',
        'database':str(a.database),'grouped_gemms':len(groups),'grouped_kernel_union_ns':compute,
        'kernel_types':{k:{'calls':len(v),'sum_ns':sum(hi-lo for lo,hi in v),'union_ns':length(v)} for k,v in kernel_types.items()},
        'weight_DMA_union_ns':length(weight_dma),'weight_DMA_fraction_overlapped':overlap/length(weight_dma) if length(weight_dma) else 0,
        'H2D_grouped_kernel_overlap_union_ns':overlap,'grouped_kernel_fraction_overlapped':overlap/compute if compute else 0,
        'device_overlap_observed':bool(overlap),'H2D_streams':rows,
        'full_prefetch_bytes_matched':a.expected_prefetch_bytes in weight_stream_bytes,
        'expected_full_prefetch_bytes':a.expected_prefetch_bytes,'examples':examples,
        'D2H_inside_grouped_window':{'copies':len(d2h),'bytes':sum(x[2] for x in d2h),'large_copies_ge1MiB':sum(x[2]>=1024*1024 for x in d2h)},
        'caveat':'Layer N+1 assignment follows accepted source schedule plus per-layer full-prefetch counters; CUDA activity proves simultaneous copy/compute, not a throughput gain by itself.'}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
