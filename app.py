from flask import Flask, render_template, request, jsonify
import sqlite3
import itertools
import random
import time
from math import comb
from queue import PriorityQueue  # 新增导入 PriorityQueue

app = Flask(__name__)

# ========== 样式配置 ==========
COLOR_PALETTE = {
    "primary": "#2A4D69",
    "secondary": "#4B86B4",
    "accent": "#DAA520",
    "background": "#F5F5F5",
    "text": "#333333"
}

FONT_CONFIG = {
    "title": ("Segoe UI", 18, "bold"),
    "subtitle": ("Segoe UI", 12),
    "body": ("Segoe UI", 10),
    "input": ("Consolas", 10)
}

# ========== 核心算法实现 ==========
def check_cover_factory(j, s):
    """根据 j, s 生成覆盖检查函数"""
    if j == s:
        return lambda perm, group: set(group).issubset(perm)
    else:
        return lambda perm, group: any(set(sub).issubset(perm) for sub in itertools.combinations(group, s))

def branch_and_bound_n_choose_s_fast(all_perms, n_elements, j, s):
    """快速分支限界算法（j == s专用）"""
    assert j == s, "此算法仅适用于j == s的情况"
    
    groups_to_cover = list(itertools.combinations(n_elements, j))
    check_cover = check_cover_factory(j, s)
    
    # 贪心预热
    _, greedy_depth, greedy_solution = greedy_heuristic_cover(all_perms.copy(), n_elements, j, s)
    best_depth = greedy_depth
    best_solution = greedy_solution
    
    # 精确搜索（当组合数较小时）
    if len(all_perms) <= 50:
        perm_covers = {}
        for perm in all_perms:
            covers = set()
            for idx, group in enumerate(groups_to_cover):
                if check_cover(perm, group):
                    covers.add(idx)
            perm_covers[perm] = covers
        
        pq = PriorityQueue()
        initial_covered = set()
        pq.put((0, 0, [], all_perms, initial_covered))
        
        while not pq.empty():
            priority, current_depth, path, perms, covered = pq.get()
            
            if current_depth >= best_depth:
                continue
                
            if len(covered) == len(groups_to_cover):
                if current_depth < best_depth:
                    best_depth = current_depth
                    best_solution = path.copy()
                continue
                
            perm_potential = []
            for perm in perms:
                newly = perm_covers[perm] - covered
                if newly:
                    perm_potential.append((len(newly), perm, newly))
            perm_potential.sort(key=lambda x: x[0], reverse=True)
            
            for _, perm, new_cover in perm_potential[:5]:
                new_path = path + [perm]
                new_perms = [p for p in perms if p != perm]
                new_covered = covered | new_cover
                pq.put((len(new_path), len(new_path), new_path, new_perms, new_covered))
    
    return (best_solution is not None), best_depth, best_solution

def greedy_heuristic_cover(all_perms, n_elements, j, s):
    """基础贪心算法实现"""
    groups_to_cover = list(itertools.combinations(n_elements, j))
    required_idx = set(range(len(groups_to_cover)))
    check_cover = check_cover_factory(j, s)
    
    perm_covers = {}
    for perm in all_perms:
        covers = set()
        for idx, group in enumerate(groups_to_cover):
            if check_cover(perm, group):
                covers.add(idx)
        perm_covers[perm] = covers
    
    selected = []
    covered = set()
    remaining_perms = all_perms.copy()
    
    while covered != required_idx:
        best_perm = max(remaining_perms, key=lambda p: len(perm_covers[p] - covered))
        selected.append(best_perm)
        covered.update(perm_covers[best_perm])
        remaining_perms.remove(best_perm)
    
    return True, len(selected), selected

def adaptive_branch_and_bound(all_perms, n_elements, j, s, n):
    """自适应算法选择器"""
    k = len(all_perms[0]) if all_perms else 0
    
    # 特例处理
    if j == s and k == j:
        return True, len(all_perms), all_perms.copy()
    
    estimated_groups = comb(n, j)
    
    if j == s:
        if estimated_groups <= 300:
            return branch_and_bound_n_choose_s_fast(all_perms, n_elements, j, s)
        elif estimated_groups <= 5000:
            return greedy_heuristic_cover(all_perms, n_elements, j, s)
        else:
            return greedy_heuristic_cover(all_perms, n_elements, j, s)
    else:
        if estimated_groups <= 1000:
            return branch_and_bound_n_choose_s_fast(all_perms, n_elements, j, s)
        else:
            return greedy_heuristic_cover(all_perms, n_elements, j, s)

# ========== 验证函数 ==========
def validate_solution_strict(n_elements, k, j, s, solution):
    """严格验证解决方案"""
    required_groups = list(itertools.combinations(n_elements, j))
    solution_sets = [set(group) for group in solution]
    
    for group in required_groups:
        group_subsets = list(itertools.combinations(group, s))
        covered = False
        for subset in group_subsets:
            subset_set = set(subset)
            for sol in solution_sets:
                if subset_set.issubset(sol):
                    covered = True
                    break
            if covered:
                break
        if not covered:
            return False
    return True

# ========== 路由 ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_n', methods=['POST'])
def generate_n():
    try:
        m = int(request.form.get('m'))
        mode = request.form.get('mode')
        n = random.randint(7, 25) if mode == "random" else int(request.form.get('n'))
        random_numbers = sorted(random.sample(range(m), n))
        formatted = " ".join(str(num+1).zfill(2) for num in random_numbers)
        return jsonify({'samples': formatted})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/execute', methods=['POST'])
def execute():
    try:
        k = int(request.form.get('k'))
        j = int(request.form.get('j'))
        s = int(request.form.get('s'))
        n_elements = request.form.get('manual_input').split()
        n = len(n_elements)
        all_perms = list(itertools.combinations(n_elements, k))

        start = time.time()
        result, depth, solution = adaptive_branch_and_bound(all_perms, n_elements, j, s, n)
        elapsed = time.time() - start

        if result:
            result_str = []
            for idx, group in enumerate(solution, 1):
                result_str.append(f"组合{idx}: {group}")
            result_str.append(f"执行时间: {elapsed:.6f}秒")
            return jsonify({'result': result_str, 'success': True})
        else:
            return jsonify({'error': '未找到解决方案', 'success': False}), 400
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/store_to_db', methods=['POST'])
def store_to_db():
    try:
        m = request.form.get('m')
        n = len(request.form.get('manual_input').split())
        k = request.form.get('k')
        j = request.form.get('j')
        s = request.form.get('s')
        solution = eval(request.form.get('solution'))
        execution_time = float(request.form.get('execution_time'))

        conn = sqlite3.connect('samples_selection.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
              filename TEXT PRIMARY KEY,
              combinations TEXT,
              execution_time REAL
            )
        ''')

        base = f"{m}-{n}-{k}-{j}-{s}"
        cursor.execute("SELECT COUNT(*) FROM results WHERE filename LIKE ?", (f"{base}%",))
        x = cursor.fetchone()[0]+1
        y = len(solution)
        filename = f"{base}-{x}-{y}"
        combinations = "\n".join(str(g) for g in solution)

        cursor.execute("INSERT INTO results (filename, combinations, execution_time) VALUES (?, ?, ?)", 
                       (filename, combinations, execution_time))
        conn.commit()
        conn.close()
        return jsonify({'message': f"保存成功！文件名: {filename}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/load_db_records', methods=['GET'])
def load_db_records():
    conn = sqlite3.connect('samples_selection.db')
    cursor = conn.cursor()
    cursor.execute("SELECT filename, execution_time, combinations FROM results")
    records = []
    for filename, time_used, combs in cursor.fetchall():
        count = len(combs.split('\n'))
        records.append({
            'filename': filename,
            'time_used': f"{time_used:.6f}",
            'count': count
        })
    conn.close()
    return jsonify({'records': records})

@app.route('/preview_record', methods=['POST'])
def preview_record():
    filename = request.form.get('filename')
    conn = sqlite3.connect('samples_selection.db')
    cursor = conn.cursor()
    cursor.execute("SELECT combinations FROM results WHERE filename=?", (filename,))
    record = cursor.fetchone()
    conn.close()
    if record:
        return jsonify({'combinations': record[0]})
    else:
        return jsonify({'error': '未找到记录'}), 400

@app.route('/validate_selected', methods=['POST'])
def validate_selected():
    filename = request.form.get('filename')
    conn = sqlite3.connect('samples_selection.db')
    cursor = conn.cursor()
    cursor.execute("SELECT combinations FROM results WHERE filename=?", (filename,))
    record = cursor.fetchone()
    conn.close()
    if record:
        try:
            groups = [eval(line) for line in record[0].split('\n')]
            all_elements = sorted(set(e for g in groups for e in g))
            tokens = filename.split("-")
            m, n, k, j, s = map(int, tokens[:5])
            if validate_solution_strict(all_elements, k, j, s, groups):
                return jsonify({'message': '✅ 验证通过！'})
            else:
                return jsonify({'message': '❌ 验证失败！'}), 400
        except Exception as e:
            return jsonify({'error': f"数据解析失败: {str(e)}"}), 400
    else:
        return jsonify({'error': '未找到记录'}), 400

@app.route('/delete_selected', methods=['POST'])
def delete_selected():
    filename = request.form.get('filename')
    conn = sqlite3.connect('samples_selection.db')
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM results WHERE filename=?", (filename,))
        conn.commit()
        conn.close()
        return jsonify({'message': '删除成功'})
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)