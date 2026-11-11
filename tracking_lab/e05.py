"""E05 executable pilot: genuine global proposals and explicit identity QA.

This is an offline Qwen/OSTrack adaptation inspired by MSRTrack, not its
MCITrack/SAM2/GPT-4.1 reproduction. Inference never reads scoring truth.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WEIGHT = '8e6de3c6f10cbc21eaf1c34532358e23618f78a1a2bb9f0d9a2e55adc7af4894'
CODE = ['tracking_lab/e05.py', 'tracking_lab/e05_qwen.py', 'tracking_lab/e05_metrics.py',
        'experiments/04_boat_identity/global_candidates.py', 'experiments/04_boat_identity/support.py']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def support():
    return module('e05_protocol_support', ROOT / 'experiments/04_boat_identity/support.py')


def slug(sequence):
    return sequence.replace('/', '_')


def overlap(a, b):
    wh = [max(0., min(a[i] + a[i+2], b[i] + b[i+2]) - max(a[i], b[i])) for i in (0, 1)]
    inter = wh[0] * wh[1]
    return inter / max(1e-12, a[2]*a[3] + b[2]*b[3] - inter)


def tile_geometry(width, height, box, cap=400):
    scale = math.sqrt(box[2] * box[3])
    if scale <= 0:
        raise ValueError('Invalid initialization area')
    side = math.ceil(scale * 5.)
    step = max(1, int(side * .9))
    centers = [(x, y) for x in range(side // 2, width + side // 2, step)
                       for y in range(side // 2, height + side // 2, step)]
    if not centers or len(centers) > cap:
        raise ValueError(f'Full-image scan requires {len(centers)} tiles, cap={cap}')
    return scale, centers


def marked_crop(rgb, box, path):
    """Same pixels for feature/generation arms; blue box is prediction, never GT."""
    from PIL import Image, ImageDraw
    height, width = rgb.shape[:2]
    x, y, w, h = [float(v) for v in box]
    if w <= 0 or h <= 0:
        raise ValueError('Invalid crop box')
    side = 1.5 * max(w, h); cx = x + w/2; cy = y + h/2
    bounds = [max(0, math.floor(cx-side/2)), max(0, math.floor(cy-side/2)),
              min(width, math.ceil(cx+side/2)), min(height, math.ceil(cy+side/2))]
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError('Empty crop')
    image = Image.fromarray(rgb).crop(tuple(bounds))
    inner = [max(0, x-bounds[0]), max(0, y-bounds[1]),
             min(image.width-1, x+w-bounds[0]), min(image.height-1, y+h-bounds[1])]
    ImageDraw.Draw(image).rectangle(inner, outline=(0, 128, 255), width=2)
    path.parent.mkdir(parents=True, exist_ok=True); image.save(path)
    return {'image': str(path.resolve()), 'sha256': sha(path), 'crop_bounds_xyxy': bounds,
            'inner_prediction_box_xyxy': inner, 'marker': 'blue predicted region; init reference only uses permitted init box'}


def plan_for(run):
    plan = read(run / 'plan.json')
    if sha(run / 'plan.json') != (run / 'plan.sha256').read_text().strip():
        raise ValueError('Frozen plan changed')
    for relative, digest in plan['code_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Code changed since prepare: ' + relative)
    if sha(Path(plan['assets'])/'ostrack/ostrack_runner.py') != plan['host_wrapper_sha256']:
        raise ValueError('Host wrapper changed since prepare')
    return plan


def prepare(args):
    from PIL import Image
    assets = args.assets.resolve(); model = args.qwen.resolve()
    manifest = read(assets / 'mvtd_exploration/inference_manifest.json')
    selected = [s for s in manifest['sequences'] if args.scope == 'all-old' or s['role'] == 'development']
    if not selected:
        raise ValueError('No selected data')
    selected.sort(key=lambda s: (s['role'] != 'development', s['sequence']))
    if sha(assets / 'ostrack/pytorch_model.bin') != EXPECTED_WEIGHT:
        raise ValueError('OSTrack checkpoint differs')
    sup = support(); sequences = []
    for s in selected:
        frames = s['frames']
        ids = [f['source_frame'] for f in frames]
        if ids != list(range(ids[0], ids[0]+len(ids))):
            raise ValueError('Noncontiguous source frames')
        for f in frames:
            path = assets / f['path']
            if not path.resolve().is_relative_to(assets) or sha(path) != f['sha256']:
                raise ValueError('Frame path/hash differs')
        with Image.open(assets / frames[0]['path']) as im:
            scale, centers = tile_geometry(im.width, im.height, s['init_bbox_xywh'])
        sequences.append({**s, 'query_indices': sup.query_indices(len(frames), cap=5),
                          'actual_global_tiles_per_query': len(centers), 'fixed_init_scale': scale})
    args.run.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable; code hashes recorded'
    plan = {'experiment': 'E05_msr_inspired_Qwen_OSTrack', 'status': 'frozen_before_inference',
            'scope': args.scope, 'independent_evaluation': False, 'assets': str(assets), 'qwen': str(model),
            'checkpoint_sha256': EXPECTED_WEIGHT,
            'host_wrapper_sha256': sha(assets/'ostrack/ostrack_runner.py'),
            'qwen_config_sha256': sha(model/'config.json') if (model/'config.json').is_file() else None,
            'local_qwen_config_available': (model/'config.json').is_file(),
            'model_revision_claim': 'record actual config/files; never substitute API or other size',
            'git_commit': commit, 'code_sha256': {p: sha(ROOT/p) for p in CODE},
            'sequences': sequences, 'max_generations': 24 if args.scope == 'development' else 48,
            'budget_definition': 'new E05 attempts, including proposer/selfcheck/direct/QA and failures; E04 remains separate',
            'candidate_rule': 'fixed initial-template/scale whole-image scan; raw responses, no Hann; NMS .5; top4 of top16',
            'query_rule': 'fixed five-or-fewer points, >=10 apart and >=5 remaining frames',
            'new_training': False, 'new_model_downloads': False, 'automatic_old_test_run': False,
            'adaptation_boundary': 'No SAM2 or original API model/trigger chain; not an MSRTrack reproduction'}
    write(args.run/'plan.json', plan)
    (args.run/'plan.sha256').write_text(sha(args.run/'plan.json')+'\n')
    print(json.dumps({'prepared': str(args.run), 'scope': args.scope, 'max_generations': plan['max_generations'],
                      'sequences': [{k:s[k] for k in ('sequence','role','query_indices','actual_global_tiles_per_query')} for s in sequences]}, indent=2))


def cache(args):
    import cv2
    import numpy as np
    import torch
    plan = plan_for(args.run); assets = Path(plan['assets'])
    output = args.run/'cache'; output.mkdir(exist_ok=False)
    sys.path.insert(0, str(assets/'ostrack'))
    from ostrack_runner import OSTrackRunner, sample_target
    helper = module('e05_fullimage_decode', ROOT/'experiments/04_boat_identity/global_candidates.py')
    torch.manual_seed(20261028); np.random.seed(20261028)
    runner = OSTrackRunner(device=args.device, checkpoint=assets/'ostrack/pytorch_model.bin')
    rows=[]; initial={}; timing=[]
    def sync():
        if args.device == 'cuda': torch.cuda.synchronize()
    def scan(rgb, init_box):
        h,w=rgb.shape[:2]; scale,centers=tile_geometry(w,h,init_box)
        boxes=[];scores=[];tiles=[];grids=[]
        sync();start=time.perf_counter()
        template_hash=hashlib.sha256(runner.template.cpu().numpy().tobytes()).hexdigest()
        for tid,(cx,cy) in enumerate(centers):
            pseudo=[cx-scale/2,cy-scale/2,scale,scale]
            patch,resize,_=sample_target(rgb,pseudo,5.,output_sz=384)
            with torch.inference_mode():
                out=runner.network(template=runner.template,search=runner._tensor(patch),ce_template_mask=runner.template_mask)
            b,s,t,g=helper.decode(out,resize,pseudo,h,w,tid)
            boxes.append(b);scores.append(s);tiles.append(t);grids.append(g)
        result=helper.nms_top(np.concatenate(boxes),np.concatenate(scores),np.concatenate(tiles),np.concatenate(grids),16)
        sync()
        assert template_hash==hashlib.sha256(runner.template.cpu().numpy().tobytes()).hexdigest()
        return result,{'tiles_executed':len(centers),'seconds':time.perf_counter()-start,'hann':False}
    for s in plan['sequences']:
        seq=s['sequence']; sid=slug(seq); history=[]; query=set(s['query_indices'])
        baseline=args.run/'baseline'/f'{sid}.jsonl';baseline.parent.mkdir(exist_ok=True)
        for fi,f in enumerate(s['frames']):
            path=assets/f['path']
            if sha(path)!=f['sha256']: raise ValueError('Frame changed')
            bgr=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if bgr is None:raise ValueError('Image failed to decode')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            sync();start=time.perf_counter()
            local=runner.initialize(rgb,s['init_bbox_xywh']) if fi==0 else runner.track(rgb)
            sync();local_seconds=time.perf_counter()-start
            with baseline.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'sequence':seq,'frame_index':fi,'source_frame':f['source_frame'],
                    'bbox_xywh':local['bbox'],'seconds':local_seconds})+'\n')
            if fi==0:
                reference=marked_crop(rgb,s['init_bbox_xywh'],args.run/'images'/sid/'reference.png')
                pool,st=scan(rgb,s['init_bbox_xywh']);negative=[]
                for c in pool:
                    if overlap(c['bbox'],s['init_bbox_xywh']) < .1:
                        # Initial box overlap is permitted. These remain POSSIBLE distractors.
                        negative.append(marked_crop(rgb,c['bbox'],args.run/'images'/sid/f'possible_distractor_{len(negative)}.png'))
                    if len(negative)==2:break
                initial[seq]={'reference':reference,'distractors':negative,'features_scope':'possible distractors, not independently confirmed identities','initial_scan':st}
                continue
            if fi not in query:continue
            pool,st=scan(rgb,s['init_bbox_xywh']);timing.append({'sequence':seq,'frame_index':fi,**st})
            if len(pool)<2:raise ValueError('Insufficient visual proposals')
            canonical=[{'bbox_xywh':c['bbox'],'tracker_score':c['score'],'original_rank':c['rank']} for c in pool[:4]]
            labeled=support().shuffled_labels(canonical,seq,f['source_frame'])
            candidates=[]
            for c in labeled:
                im=marked_crop(rgb,c['bbox_xywh'],args.run/'images'/sid/f'{fi:06d}_{c["label"]}.png')
                candidates.append({**c,**im})
            rows.append({'case_id':f'{sid}_{fi:06d}','sequence':seq,'role':s['role'],'frame_index':fi,
                         'source_frame':f['source_frame'],'local_bbox_xywh':list(local['bbox']),
                         'reference':reference,'history':list(history),'distractors':negative,'candidates':candidates,
                         'global_top16':pool,'global_scan':st})
            hist=marked_crop(rgb,local['bbox'],args.run/'images'/sid/f'{fi:06d}_past_local.png')
            history=(history+[{'source_frame':f['source_frame'],**hist}])[-2:]
        assert fi+1==len(s['frames'])
    write(output/'candidates.json',{'rows':rows,'initial':initial,'global_timing':timing,
          'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU',
          'torch':torch.__version__,'inference_reads_evaluation_file':False,'complete_baseline_frames':True})
    print(json.dumps({'cached_queries':len(rows),'global_forward_count':sum(t['tiles_executed'] for t in timing),
                      'global_candidates_not_local_head':True,'no_generations':True},indent=2))


def inspect(args):
    plan=plan_for(args.run); data=read(args.run/'cache/candidates.json')
    truth=read(Path(plan['assets'])/'mvtd_exploration/evaluation.json')
    gt={s['sequence']:{f['local_index']:f for f in s['frames']} for s in truth['sequences']}
    stats={}
    for role in sorted({r['role'] for r in data['rows']}):
        counts=defaultdict(int)
        for r in data['rows']:
            if r['role']!=role:continue
            counts['queries']+=1;g=gt[r['sequence']][r['frame_index']+1]
            if not g['bbox_valid']:counts['unknown']+=1;continue
            counts['valid']+=1;box=g['bbox_xywh']
            recall=any(overlap(c['bbox_xywh'],box)>=.5 for c in r['candidates'])
            native=min(r['candidates'],key=lambda c:c['original_rank'])
            top=overlap(native['bbox_xywh'],box)>=.5;local=overlap(r['local_bbox_xywh'],box)>=.5
            counts['recall4']+=recall;counts['native_top1_success']+=top;counts['local_success']+=local
            counts['candidate_ranking_headroom']+=bool(recall and not top)
            counts['actual_local_recovery_headroom']+=bool(recall and not local)
        stats[role]=dict(counts)
    result={'scope':'CPU candidate opportunity check, not a language-model result','by_role':stats,
            'development_has_ranking_space':stats.get('development',{}).get('candidate_ranking_headroom',0)>0,
            'automatic_evaluation_authorized':False,'no_query_selection_based_on_truth':True}
    write(args.run/'opportunity.json',result);print(json.dumps(result,indent=2))


def vlm(args):
    from tracking_lab.e05_qwen import E05Qwen
    plan=plan_for(args.run);data=read(args.run/'cache/candidates.json')
    opportunities=read(args.run/'opportunity.json')
    if not opportunities['development_has_ranking_space'] and not args.interface_smoke and plan['scope']!='all-old':
        raise ValueError('Development has no top1 correction room; use --interface-smoke for bounded interface checks, not a gain claim')
    model_config=Path(plan['qwen'])/'config.json'
    if not model_config.is_file():
        raise FileNotFoundError('Qwen config missing; supply an existing local model at prepare')
    if plan['qwen_config_sha256'] and sha(model_config)!=plan['qwen_config_sha256']:
        raise ValueError('Qwen config changed since prepare')
    output=args.run/'inference';output.mkdir(exist_ok=False)
    runner=E05Qwen(plan['qwen'], output/'model', max_calls=12 if args.interface_smoke else plan['max_generations'],device=args.device)
    started=time.perf_counter()
    runner.load();rows=[];attributes={}
    for seq, initial in data['initial'].items():
        seqrows=[r for r in data['rows'] if r['sequence']==seq]
        if args.interface_smoke:
            seqrows=[r for r in seqrows if r['role']=='development'][:2]
        if not seqrows:
            continue
        reference=initial['reference']['image']; negatives=[x['image'] for x in initial['distractors']]
        proposed=runner.propose(reference, negatives, case_id=slug(seq)+'_propose')
        features=proposed.get('features',[])
        checked=None
        if features:
            checked=runner.choose(reference,[],[{'label':'A','image':reference}],features=features,
                mode='contrastive',case_id=slug(seq)+'_selfcheck',distractors=negatives)
            table=checked.get('support_table',{})
            answers=table.get('A',[]) if isinstance(table,dict) else []
            features=[feature for feature,answer in zip(features,answers) if answer=='supported']
        attributes[seq]={'proposed':proposed,'selfcheck':checked,'features':features}
        write(output/'attributes.json',attributes)
        for r in seqrows:
            history=[x['image'] for x in r['history']]
            candidates=[{'label':c['label'],'image':c['image']} for c in r['candidates']]
            visual=runner.visual_scores(reference,history,candidates,distractors=negatives)
            direct=runner.choose(reference,history,candidates,features=features,mode='direct',case_id=r['case_id']+'_direct',distractors=negatives)
            qa=runner.choose(reference,history,candidates,features=features,mode='contrastive',case_id=r['case_id']+'_qa',distractors=negatives)
            row={**r,'visual_scores':visual['scores'],'contrastive_visual_scores':visual['contrastive_scores'],
                 'visual_metadata':visual,'direct':direct,'qa':qa,'frozen_features':features}
            rows.append(row)
            with (output/'predictions.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(f'E05 model phase completed {seq}, queries={len(seqrows)}',flush=True)
    write(output/'completed.json',{'rows':len(rows),'interface_smoke_only':args.interface_smoke,'no_closedloop':True,
          'generation_attempts':runner.calls_used,
          'visual_encoder_calls':sum(r['visual_metadata']['image_encoder_calls'] for r in rows),
          'model_phase_elapsed_seconds':time.perf_counter()-started,
          'plan_sha256':sha(args.run/'plan.json'),'candidate_cache_sha256':sha(args.run/'cache/candidates.json')})


def score(args):
    from tracking_lab.e05_metrics import calibrate, measure
    plan=plan_for(args.run)
    rows=[json.loads(line) for line in (args.run/'inference/predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    truth=read(Path(plan['assets'])/'mvtd_exploration/evaluation.json')
    settings=calibrate(rows,truth,manifest_sha256=sha(args.run/'cache/candidates.json'))
    report=measure(rows,truth,settings)
    comparable=[r for r in report['rows'] if r['state']=='annotated' and
                r['arms']['qa']['attempted'] and r['arms']['direct']['attempted']]
    report['qa_vs_direct']={
        'paired_valid_n':len(comparable),
        'fixes':sum(r['arms']['qa']['applied_success'] and not r['arms']['direct']['applied_success'] for r in comparable),
        'damages':sum(not r['arms']['qa']['applied_success'] and r['arms']['direct']['applied_success'] for r in comparable)}
    report['qa_vs_direct']['net']=report['qa_vs_direct']['fixes']-report['qa_vs_direct']['damages']
    completed=read(args.run/'inference/completed.json')
    report['execution']=completed
    write(args.run/'settings_development_only.json',settings);write(args.run/'report.json',report)
    lines=['# E05 开发机制验证结果','',f"模式：{plan['scope']}；独立评价：否；接口冒烟：{completed['interface_smoke_only']}。",'',
       f"实际生成尝试{completed['generation_attempts']}次（含提议、自检、失败）；视觉编码{completed['visual_encoder_calls']}次；模型阶段耗时{completed['model_phase_elapsed_seconds']:.2f}秒（含加载与IO）。",'',
       '本轮为MSRTrack启发的Qwen/OSTrack适配，不含SAM2与原API模型；未运行闭环。',
       '逐序列、UNKNOWN、候选上限、原生rank1、视觉及QA对照见report.json。',
       '若QA优于直接判断但不优于原局部/主视觉，只能称减少错误介入，不能称提高跟踪恢复。','',
       '开发投入信号（不自动授权评价）：','```json',json.dumps(report.get('development_signal'),ensure_ascii=False,indent=2),'```','',
       '实际原始消息、逐图角色、图片hash、支持表及生成次数见inference/model；完整基线和真全图候选见baseline与cache。']
    lines.extend(['','## 实际对照（开发与旧探索片段分列）','',
        '候选选对与采纳后的定位成功分别统计；拒答保留当前局部框，不能算作成功身份判断。',
        'NOT_RUN不进入该模型分母，不能与完整覆盖方法直接比率排序。阈值及主视觉B仅按开发数据选择。',''])
    for role, group in report['by_role'].items():
        lines.extend([f'### {role}','',
            f"查询{group['queries']}；有效定位{group['valid_localization_n']}；UNKNOWN {group['unknown_n']}；前4候选可达{group['recall_at_k_count']}；原生第一候选成功{group['native_top1_success_count']}。",'',
            '| 方法 | 实际执行 | 有效定位分母 | 选对候选 | 应用后定位成功 | 错误接受 | 拒答 | 无效输出 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for arm in ('local','B0','B1','B2','direct','qa'):
            a=group['arms'][arm]
            lines.append(f"| {arm} | {a['attempted']} | {a['valid_localization_n']} | {a['candidate_choice_success_count']} | {a['applied_success_count']} | {a['wrong_accept']} | {a['rejected']} | {a['invalid']} |")
        q=group['paired']['qa']
        lines.extend(['',f"QA对主视觉B：{json.dumps(q['vs_main_b'],ensure_ascii=False)}",'',
                      f"QA对局部跟踪：{json.dumps(q['vs_local'],ensure_ascii=False)}",''])
    lines.extend(['','QA对直接判断（仅共同有效查询）：'+json.dumps(report['qa_vs_direct'],ensure_ascii=False),
                  '', '全体查询图片可用 `python -m tracking_lab.e05_visualize --run 本次目录` 导出；图中GT/IoU仅供事后评分，未输入模型。'])
    (args.run/'REPORT_中文.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(args.run/'report.json'),'main_b':settings.get('main_b'),
                      'development_signal':report.get('development_signal')},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','cache','inspect','vlm','score'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--assets',type=Path,default=ROOT/'assets/e04')
    p.add_argument('--qwen',type=Path,default=ROOT/'models/qwen25-7b');p.add_argument('--scope',choices=['development','all-old'],default='development')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--interface-smoke',action='store_true');a=p.parse_args()
    a.run=a.run.resolve()
    import os
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    {'prepare':prepare,'cache':cache,'inspect':inspect,'vlm':vlm,'score':score}[a.command](a)


if __name__=='__main__':main()
