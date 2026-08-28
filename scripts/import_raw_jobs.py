#!/usr/bin/env python3
"""把 scraper 导出的行业 Excel 批量导入 raw_jobs 表。

Excel 列需与 raw_jobs 字段一致（不含自增 id）：
  title | salary_text | city | education | experience | requirements | company | source

用法（必须在仓库根目录运行，确保 load_dotenv 找到 .env）:
  python scripts/import_raw_jobs.py --dir <excel目录> --dry-run
  python scripts/import_raw_jobs.py --dir <excel目录>              # 追加导入
  python scripts/import_raw_jobs.py --dir <excel目录> --reset      # 清空 raw_jobs 后导入
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal
from backend.models import RawJob

COLUMNS = ["title", "salary_text", "city", "education", "experience",
           "requirements", "company", "source"]


def _clean(v):
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def load_excel(path):
    df = pd.read_excel(path)
    # 表头形如 "title\n岗位名称"，取换行前部分作为字段名
    df.columns = [str(c).split("\n")[0] for c in df.columns]
    return df


def main():
    p = argparse.ArgumentParser(description="Excel 批量导入 raw_jobs 表")
    p.add_argument("--dir", required=True, help="行业 Excel 所在目录")
    p.add_argument("--reset", action="store_true", help="导入前清空 raw_jobs 表")
    p.add_argument("--dry-run", action="store_true", help="只统计不写入")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.xlsx")))
    if not files:
        print(f"目录下没有 .xlsx 文件: {args.dir}")
        return 1

    db = SessionLocal()
    total = 0
    try:
        if args.reset:
            n = db.query(RawJob).delete()
            db.commit()
            print(f"已清空 raw_jobs 表（删除 {n} 条）\n")

        for f in files:
            df = load_excel(f)
            n = len(df)
            print(f"  {os.path.basename(f):24s} {n:>6} 条")
            if not args.dry_run:
                for _, row in df.iterrows():
                    db.add(RawJob(
                        title=_clean(row.get("title")),
                        salary_text=_clean(row.get("salary_text")),
                        city=_clean(row.get("city")),
                        education=_clean(row.get("education")),
                        experience=_clean(row.get("experience")),
                        requirements=_clean(row.get("requirements")),
                        company=_clean(row.get("company")),
                        source=_clean(row.get("source")),
                    ))
                db.commit()
            total += n
    finally:
        db.close()

    print(f"\n{'（dry-run，未写入）' if args.dry_run else '导入完成'} 共 {total} 条 → raw_jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
