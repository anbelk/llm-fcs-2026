"""Create a brief PDF from actual run artifacts; absent runs stay explicitly absent."""
import argparse
from datetime import datetime,timezone
from html import escape
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--font',default='/System/Library/Fonts/Supplemental/Arial.ttf')
    ap.add_argument('--bold-font',default='/System/Library/Fonts/Supplemental/Arial Bold.ttf')
    args=ap.parse_args()
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
    from reportlab.lib.pagesizes import A4
    pdfmetrics.registerFont(TTFont('Body',args.font));pdfmetrics.registerFont(TTFont('BodyBold',args.bold_font))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TextRu',fontName='Body',fontSize=10,leading=14,spaceAfter=8))
    styles.add(ParagraphStyle(name='TitleRu',fontName='BodyBold',fontSize=18,leading=23,spaceAfter=13))
    styles.add(ParagraphStyle(name='HeadRu',fontName='BodyBold',fontSize=11,leading=15,spaceBefore=8,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle(name='SmallRu',fontName='Body',fontSize=8,leading=11,spaceAfter=7))
    story=[]
    def para(s,style='TextRu'):
        for number in ('11 300','10 731','1 165'):
            s=s.replace(number,number.replace(' ', '\u00a0'))
        story.append(Paragraph(escape(s),styles[style]))
    para('ДЗ5. RL для вызова инструментов','TitleRu')
    para('Андрей Белкин | Qwen/Qwen3.5-2B | 12 ноября 2026','SmallRu')
    para('Данные и постановка','HeadRu')
    para('RL выполняется только на Team-ACE/ToolACE, без SFT. Исходная модель закреплена ревизией 15852e8c16360a2fea060d615a32b45270f8a8fc. Из 11 300 исходных диалогов подготовлено 10 731 train и 1 165 validation. Диалоги и одинаковые prompts не пересекают части; 500 нестандартных диалогов, 6 спорных эталонов и 14 длинных примеров отложены. Продолжения после tool в обучение не входят.')
    para('В BFCL найдено 0 точных совпадений нормализованных пользовательских реплик; семантические утечки не исключены. BFCL не использовался для выбора шага или настройки обучения.')
    para('Награды и одинаковые настройки','HeadRu')
    para('A: strict, KL 0.02. B: partial, KL 0.02. C: partial, KL 0.10. Strict проверяет мультимножество вызовов с аргументами. В partial совпавшее имя получает 0.25 + 0.75 × F1 аргументов; вызовы сопоставляются один к одному, результат равен 2 × суммарная оценка / (число предсказанных + эталонных вызовов). Повторы сохраняются; вложенное значение сравнивается целиком. 1 и 1.0 равны, bool и строки отличаются от чисел. Optional/default не нормализуются. Пустой, обрезанный и сломанный ответ получает 0; для no-call нужен непустой ответ без вызовов.')
    cfg_path=args.runs/'A_strict/training/config.json'
    if cfg_path.exists():
        cfg=json.loads(cfg_path.read_text())
        para(f"GRPO + LoRA r={cfg['lora_r']}, alpha={cfg['lora_alpha']}; BF16, training/data seed 42 (rollout vLLM seed 0), LR {cfg['learning_rate']}, {cfg['max_steps']} optimizer steps. На шаг 8 rollout-ответов (4 на prompt), то есть 2 исходных prompts. Выбор лучшего шага - по common strict на фиксированных {cfg['validation_selection_size']} примерах ToolACE validation. Итоговая validation оценивается целиком. Перед BFCL адаптер объединяется с исходными весами.")
    else:para('Обучение ещё не сформировало сохранённую конфигурацию. Плановые настройки не являются результатом запуска.')
    para('Фактические результаты','HeadRu')
    comparison_path=args.runs/'comparison.json'
    comparison=json.loads(comparison_path.read_text()) if comparison_path.exists() else []
    rows=[['Вариант','BFCL overall','Val strict','Val partial']]
    for name in ('baseline','A_strict','B_partial','C_partial_kl'):
        r=next((r for r in comparison if r['run']==name),{})
        fmt=lambda key:f"{r[key]:.3f}" if key in r else 'не выполнено'
        rows.append([name,fmt('overall'),fmt('validation_strict'),fmt('validation_partial')])
    table=Table(rows,colWidths=[135,116,110,110],repeatRows=1)
    table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Body'),('FONTNAME',(0,0),(-1,0),'BodyBold'),
        ('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff5')),
        ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#b6c1cc')),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8)]))
    story.append(table);story.append(Spacer(1,9))
    para('BFCL overall усредняет группы и подкатегории, поэтому не равен просто correct/350. Evaluator и benchmark неизменны: thinking off, temperature 0, seed 42, 1024 новых токена, контекст 4096. Файлы metrics.json и predictions.jsonl сохраняются отдельно для каждого завершённого варианта.','SmallRu')
    chart=args.runs/'loss_reward.png'
    if chart.exists():
        story.append(Image(str(chart),width=470,height=174))
        para('Loss относится к GRPO. Reward у strict и partial определяется по-разному; высоты кривых разных наград напрямую не сравниваются.','SmallRu')
    errors_path=args.runs/'error_analysis.json'
    if errors_path.exists():
        analysis=json.loads(errors_path.read_text());para('Изменения и ошибки','HeadRu')
        group_rows=[['Группа BFCL','Base','A','B','C']]
        names=('baseline','A_strict','B_partial','C_partial_kl')
        for group in ('simple','multiple','parallel','parallel_multiple','irrelevance'):
            vals=[]
            for name in names:
                path=args.runs/name/'bfcl/metrics.json'
                vals.append(f"{json.loads(path.read_text())['groups'][group]:.3f}" if path.exists() else '-')
            group_rows.append([group,*vals])
        gt=Table(group_rows,colWidths=[175,74,74,74,74],repeatRows=1)
        gt.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Body'),('FONTNAME',(0,0),(-1,0),'BodyBold'),('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff5')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#b6c1cc')),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(gt);story.append(Spacer(1,8))
        for name in ('A_strict','B_partial','C_partial_kl'):
            if name not in analysis:continue
            counts=analysis[name]['outcomes_vs_baseline']
            para(f"{name}: исправлено относительно baseline {counts.get('improved',0)}, новых ошибок {counts.get('regressed',0)}, осталось неверных {counts.get('still_wrong',0)}.")
        signals=[(name,item['training_signal'])for name,item in analysis.items()if 'training_signal'in item]
        if signals:
            para('Группы с нулевым advantage: '+ '; '.join(f"{name}: {s['zero_advantage_groups']}/{s['prompt_groups']} ({s['zero_advantage_group_fraction']:.1%})"for name,s in signals)+'. В таких группах все rollout-награды одинаковы; относительный policy advantage равен нулю, хотя KL может продолжать менять веса. Это ограничивает полезный сигнал короткого запуска.')
        available=[r for r in comparison if r.get('run')!='baseline' and 'overall'in r]
        if available:
            name=max(available,key=lambda r:r['overall'])['run']
            categories=analysis.get(name,{}).get('first_example_per_category_and_outcome',{})
            for kind,label in [('improved','Исправление'),('regressed','Регрессия'),('still_wrong','Оставшаяся ошибка')]:
                found=categories.get(kind,{})
                selected=next((found[c]for c in ('parallel','simple','multiple','parallel_multiple','live_multiple','irrelevance','live_irrelevance')if c in found),None)
                for ex in [selected] if selected else []:
                    before,after=ex['baseline_calls'],ex['calls']
                    if not before and after:
                        detail=f"{label}, {ex['id']}: baseline не выдал разобранных вызовов; {name} вернул {len(after)} корректных вызова."
                    elif not before and not after:
                        detail=f"{label}, {ex['id']}: обе модели не выдали ожидаемый вызов инструмента."
                    else:
                        extra=[];missing=[]
                        for c in after:
                            for fn,argv in c.items():
                                old=next((b[fn]for b in before if fn in b),{})
                                if isinstance(argv,dict)and isinstance(old,dict):
                                    extra.extend(f'{k}={v!r}'for k,v in argv.items()if k not in old)
                                    missing.extend(f'{k}={v!r}'for k,v in old.items()if k not in argv)
                        reason=(f"{name} добавил {', '.join(extra)}" if extra else f"{name} пропустил {', '.join(missing)}" if missing else '')
                        detail=f"{label}, {ex['id']}: "+(reason+", и правильный ответ baseline стал ошибкой." if reason else f"Результат проверки: {json.dumps(ex['check'],ensure_ascii=False)[:220]}")
                    para(detail,'SmallRu')
    para('Ограничения и воспроизводимость','HeadRu')
    para('Каждая конфигурация использовала около 64 исходных prompts, то есть 0.6% train, и один seed. Короткий бюджет RL и небольшой учебный BFCL не дают оснований для широких выводов. Содержимое свободных ответов no-call отдельно не оценивается. EDA, разбиение, награды, configs, логи, выбранные чекпойнты, версии, хеши данных и разбор ошибок сохранены в hw5_rl; отсутствующие результаты явно отмечены выше. Метрики вычислены только по фактическим предсказаниям.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('Body',8);canvas.drawString(42,25,'LLM / домашнее задание 5');canvas.drawRightString(A4[0]-42,25,str(doc.page))
    doc=SimpleDocTemplate(str(args.output),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=42)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':main()
