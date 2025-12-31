from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from config import cfg


def _get_brics_fragments(mol, original_bonds_info, num_original_atoms):
    print("\n  [BRICS_HELPER] === 开始 BRICS 拆分 ===")
    fragments_list = []
    added_fragments_signatures = set()

    brics_bonds_info = list(BRICS.FindBRICSBonds(mol))
    if not brics_bonds_info:
        print("  [BRICS_HELPER] 未找到可用 BRICS 方法断裂的键。")
        return fragments_list

    try:
        brics_bond_indices = [mol.GetBondBetweenAtoms(i[0][0], i[0][1]).GetIdx() for i in brics_bonds_info]
        print(f"  [BRICS_HELPER] 找到待切割 BRICS 键索引 ({len(brics_bond_indices)}个): {sorted(brics_bond_indices)}")
    except Exception as e:
        print(f"  [BRICS_HELPER ERROR] 从 BRICS 结果提取键索引时出错: {e}")
        return fragments_list

    print("  [BRICS_HELPER] 执行 FragmentOnBonds (addDummies=False)...")
    frag_atoms_tuples = []
    try:
        fragmented_mol = Chem.FragmentOnBonds(mol, brics_bond_indices, addDummies=False)
        frag_atoms_tuples = Chem.GetMolFrags(fragmented_mol, asMols=False, sanitizeFrags=True)
        print(
            f"  [BRICS_HELPER] Chem.GetMolFrags (sanitize=True) 返回了 {len(frag_atoms_tuples)} 个 BRICS 片段的原子索引元组。")
    except Exception as e_frag:
        print(f"  [BRICS_HELPER ERROR] BRICS FragmentOnBonds 或 GetMolFrags (sanitize=True) 出错: {e_frag}")
        try:
            print("  [BRICS_HELPER WARN] 尝试不 Sanitize 再次获取...")
            if 'fragmented_mol' not in locals(): 
                print("  [BRICS_HELPER ERROR] fragmented_mol 未定义，无法尝试不 sanitize 获取。")
                return fragments_list
            frag_atoms_tuples = Chem.GetMolFrags(fragmented_mol, asMols=False, sanitizeFrags=False)
            print(f"  [BRICS_HELPER] Chem.GetMolFrags (sanitize=False) 返回 {len(frag_atoms_tuples)} 个片段。")
        except Exception as e_frag2:
            print(f"  [BRICS_HELPER ERROR] 再次获取仍失败: {e_frag2}")
            return fragments_list

    print(f"  [BRICS_HELPER] 开始处理 {len(frag_atoms_tuples)} 个 BRICS 片段...")
    for i, atom_indices_tuple in enumerate(frag_atoms_tuples):
        print(f"    [BRICS_HELPER DETAIL] --- 处理 BRICS 片段 {i + 1}/{len(frag_atoms_tuples)} ---")
        print(f"      [BRICS_HELPER DETAIL] 原始原子索引元组: {atom_indices_tuple}")

        current_frag_original_atoms = list(atom_indices_tuple) 

        valid_indices = all(
            isinstance(idx, int) and 0 <= idx < num_original_atoms for idx in current_frag_original_atoms)
        if not valid_indices or not current_frag_original_atoms: 
            print(f"      [BRICS_HELPER WARN] 片段 {i + 1} 因无效索引或为空而被跳过。")
            continue

        current_frag_atom_set = set(current_frag_original_atoms)
        current_frag_original_bonds = []
        for bond_idx, (a1, a2) in original_bonds_info.items():
            if a1 in current_frag_atom_set and a2 in current_frag_atom_set:
                current_frag_original_bonds.append(bond_idx)
        print(
            f"      [BRICS_HELPER DETAIL] 计算得到内部键 ({len(current_frag_original_bonds)}个): {sorted(current_frag_original_bonds)}")

        current_frag_original_atoms.sort()
        current_frag_original_bonds.sort()

        fragment_signature = (frozenset(current_frag_original_atoms), frozenset(current_frag_original_bonds))
        if fragment_signature not in added_fragments_signatures:
            added_fragments_signatures.add(fragment_signature)
            fragments_list.append([current_frag_original_atoms, current_frag_original_bonds])
            print(
                f"      [BRICS_HELPER DETAIL] 添加有效片段 (原子数 {len(current_frag_original_atoms)}, 键数 {len(current_frag_original_bonds)}) 到列表。")
        else:
            print(f"      [BRICS_HELPER INFO] 片段 {i + 1} 与此方法内已添加的片段重复，跳过。")

    print(f"  [BRICS_HELPER] === BRICS 拆分结束，共生成 {len(fragments_list)} 个独特片段 ===")
    return fragments_list


def _get_murcko_fragments(mol, original_bonds_info, num_original_atoms):
    """执行Murcko骨架拆分，返回片段列表"""
    print("\n  [MURCKO_HELPER] === 开始 Murcko 骨架拆分 ===")
    fragments_list = []
    added_fragments_signatures = set()

    print("  [MURCKO_HELPER] 步骤 1: 识别 Murcko 骨架...")
    scaffold_mol_obj = None 
    scaffold_atom_indices_set = set()
    try:
        scaffold_mol_obj = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold_mol_obj and scaffold_mol_obj.GetNumAtoms() > 0:
            print(
                f"    [MURCKO_HELPER DETAIL] 找到 Murcko 骨架 (原子数: {scaffold_mol_obj.GetNumAtoms()}, SMILES: {Chem.MolToSmiles(scaffold_mol_obj)})。")
            print(f"    [MURCKO_HELPER DETAIL] 在原始分子中匹配骨架以获取原子索引...")
            matches = mol.GetSubstructMatches(scaffold_mol_obj, uniquify=True)
            if matches: 
                if len(matches) > 1: print(
                    f"    [MURCKO_HELPER WARN] Murcko 骨架找到 {len(matches)} 个匹配项，使用第一个。")
                scaffold_atom_indices_set = set(matches[0]) 
                print(
                    f"    [MURCKO_HELPER DETAIL] 获取到骨架原始原子索引集合 (大小: {len(scaffold_atom_indices_set)})。")
            else:
                print(f"    [MURCKO_HELPER WARN] 未能在原始分子中匹配到计算出的 Murcko 骨架！这通常不应发生。")
                scaffold_atom_indices_set = set() 
        else:
            print("    [MURCKO_HELPER INFO] 分子不包含 Murcko 骨架或骨架为空。")
            return fragments_list 
    except Exception as e_scaffold:
        print(f"    [MURCKO_HELPER ERROR] 获取或匹配 Murcko 骨架时出错: {e_scaffold}")
        return fragments_list 

    if scaffold_atom_indices_set and len(scaffold_atom_indices_set) < num_original_atoms: 
        scaffold_atoms_list = sorted(list(scaffold_atom_indices_set))
        scaffold_bonds_list = []
        for bond_idx, (a1, a2) in original_bonds_info.items():
            if a1 in scaffold_atom_indices_set and a2 in scaffold_atom_indices_set:
                scaffold_bonds_list.append(bond_idx)
        scaffold_bonds_list.sort()

        scaffold_signature = (frozenset(scaffold_atoms_list), frozenset(scaffold_bonds_list))
        if scaffold_signature not in added_fragments_signatures:
            added_fragments_signatures.add(scaffold_signature)
            fragments_list.append([scaffold_atoms_list, scaffold_bonds_list])
            print(
                f"    [MURCKO_HELPER DETAIL] 添加 Murcko 骨架自身作为片段 (原子数 {len(scaffold_atoms_list)}, 键数 {len(scaffold_bonds_list)})。")

    print("  [MURCKO_HELPER] 步骤 2: 识别骨架与侧链的连接键...")
    bonds_to_cut_indices = set()
    if scaffold_atom_indices_set: 
        for bond in mol.GetBonds():
            a1_idx = bond.GetBeginAtomIdx()
            a2_idx = bond.GetEndAtomIdx()
            if (a1_idx in scaffold_atom_indices_set) != (a2_idx in scaffold_atom_indices_set):
                bonds_to_cut_indices.add(bond.GetIdx())
        print(f"    [MURCKO_HELPER DETAIL] 找到 {len(bonds_to_cut_indices)} 个骨架-侧链连接键。")
    else: 
        print("    [MURCKO_HELPER INFO] 未找到骨架原子，无法识别连接键进行侧链切割。")
        print(f"  [MURCKO_HELPER] === Murcko 骨架拆分结束（无侧链切割），共生成 {len(fragments_list)} 个片段 ===")
        return fragments_list

    sorted_bonds_to_cut_list = sorted(list(bonds_to_cut_indices))
    print(f"  [MURCKO_HELPER] 最终待切割键索引 ({len(sorted_bonds_to_cut_list)}个): {sorted_bonds_to_cut_list}")

    if not sorted_bonds_to_cut_list:
        print("  [MURCKO_HELPER INFO] 未找到可切割的骨架-侧链连接键，不进行进一步片段化。")
        print(f"  [MURCKO_HELPER] === Murcko 骨架拆分结束（无侧链切割），共生成 {len(fragments_list)} 个片段 ===")
        return fragments_list

    print("  [MURCKO_HELPER] 步骤 3: 调用 Chem.FragmentOnBonds (获取侧链)...")
    frag_atoms_tuples = []
    try:
        fragmented_mol_sidechains = Chem.FragmentOnBonds(mol, sorted_bonds_to_cut_list, addDummies=False)
        print("  [MURCKO_HELPER] Chem.FragmentOnBonds 调用完成。")
        print("  [MURCKO_HELPER] 步骤 4: 调用 Chem.GetMolFrags (获取侧链)...")
        frag_atoms_tuples = Chem.GetMolFrags(fragmented_mol_sidechains, asMols=False, sanitizeFrags=True)
        print(
            f"  [MURCKO_HELPER] Chem.GetMolFrags (sanitize=True) 返回 {len(frag_atoms_tuples)} 个片段（可能包含骨架和侧链）。")
    except Exception as e_fob_murcko:
        print(f"  [MURCKO_HELPER ERROR] Murcko FragmentOnBonds 或 GetMolFrags (sanitize=True) 出错: {e_fob_murcko}")
        try:
            print("  [MURCKO_HELPER WARN] 尝试不 Sanitize 再次获取...")
            if 'fragmented_mol_sidechains' not in locals():
                print("  [MURCKO_HELPER ERROR] fragmented_mol_sidechains 未定义，无法尝试不 sanitize 获取。")
                return fragments_list
            frag_atoms_tuples = Chem.GetMolFrags(fragmented_mol_sidechains, asMols=False, sanitizeFrags=False)
            print(f"  [MURCKO_HELPER] Chem.GetMolFrags (sanitize=False) 返回 {len(frag_atoms_tuples)} 个片段。")
        except Exception as e_fob_murcko2:
            print(f"  [MURCKO_HELPER ERROR] 再次获取仍失败: {e_fob_murcko2}")
            return fragments_list  

    print(f"  [MURCKO_HELPER] 步骤 5 & 6: 开始处理 {len(frag_atoms_tuples)} 个潜在的骨架/侧链片段...")
    for i, atom_indices_tuple in enumerate(frag_atoms_tuples):
        print(f"    [MURCKO_HELPER DETAIL] --- 处理Murcko产生的片段 {i + 1}/{len(frag_atoms_tuples)} ---")
        print(f"      [MURCKO_HELPER DETAIL] 原始原子索引元组: {atom_indices_tuple}")

        current_frag_atoms = list(atom_indices_tuple)
        valid = all(isinstance(idx, int) and 0 <= idx < num_original_atoms for idx in current_frag_atoms)
        if not valid or not current_frag_atoms:
            print(f"      [MURCKO_HELPER WARN] 此片段因无效索引或为空而被跳过。")
            continue

        current_frag_atom_set = set(current_frag_atoms)
        current_frag_bonds = []
        for bond_idx, (a1, a2) in original_bonds_info.items():
            if a1 in current_frag_atom_set and a2 in current_frag_atom_set:
                current_frag_bonds.append(bond_idx)
        print(
            f"      [MURCKO_HELPER DETAIL] 计算得到内部键 ({len(current_frag_bonds)}个): {sorted(current_frag_bonds)}")

        current_frag_atoms.sort()
        current_frag_bonds.sort()

        fragment_signature = (frozenset(current_frag_atoms), frozenset(current_frag_bonds))
        if fragment_signature not in added_fragments_signatures:
            added_fragments_signatures.add(fragment_signature)
            fragments_list.append([current_frag_atoms, current_frag_bonds])
            print(
                f"      [MURCKO_HELPER DETAIL] 添加有效片段 (原子数 {len(current_frag_atoms)}, 键数 {len(current_frag_bonds)}) 到列表。")
        else:
            print(f"      [MURCKO_HELPER INFO] 片段 {i + 1} 与此方法内已添加的片段重复，跳过。")

    print(f"  [MURCKO_HELPER] === Murcko 骨架拆分结束，共生成 {len(fragments_list)} 个独特片段 ===")
    return fragments_list


def load_smarts_from_final_fg_txt(file_path):
    """从最终筛选和处理后的官能团TXT文件加载SMARTS定义"""
    print(f"\n[LOADER] === 开始从SMARTS文件加载定义: {file_path} ===")
    smarts_dict = {}
    raw_smarts_count = 0
    loaded_smarts_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                stripped_line = line.strip()
                is_header_comment_or_empty = False
                if not stripped_line or stripped_line.startswith('#'):
                    header_keywords = [
                        "# Processed SMARTS Patterns", "# Automatically generated on:",
                        "# Total entries:", "# Format: Name: SMARTS_string",
                        "# Note: SMARTS definitions were filtered", "################################"
                    ]
                    if not stripped_line or any(keyword in stripped_line for keyword in header_keywords):
                        is_header_comment_or_empty = True
                if is_header_comment_or_empty: continue
                if ':' in stripped_line:
                    parts = stripped_line.split(':', 1)
                    name = parts[0].strip()
                    smarts = parts[1].strip().split(' #', 1)[0].strip()
                    raw_smarts_count += 1
                    if name and smarts:
                        query_mol_check = Chem.MolFromSmarts(smarts)
                        if query_mol_check is None:
                            print(f"  [LOADER WARN] 第 {line_number} 行: 无效SMARTS，跳过. '{name}' -> '{smarts}'")
                            continue
                        if len(Chem.GetMolFrags(query_mol_check)) > 1:
                            print(f"  [LOADER WARN] 第 {line_number} 行: 非连通SMARTS，跳过. '{name}' -> '{smarts}'")
                            continue
                        if name in smarts_dict and smarts_dict[name] != smarts:
                            print(f"  [LOADER WARN] 第 {line_number} 行: 重复名称 '{name}' 但SMARTS不同. 用新的.")
                        smarts_dict[name] = smarts
                        loaded_smarts_count += 1
                    else:
                        print(f"  [LOADER WARN] 第 {line_number} 行: 空名称或SMARTS: '{stripped_line}'")
                else:
                    print(f"  [LOADER INFO] 第 {line_number} 行: 跳过格式不符: '{stripped_line}'")
    except FileNotFoundError:
        print(f"[LOADER ERROR] SMARTS文件未找到: {file_path}"); return {}
    except Exception as e:
        print(f"[LOADER ERROR] 加载SMARTS文件 '{file_path}' 时出错: {e}")
    print(f"[LOADER INFO] 原始SMARTS候选条目数: {raw_smarts_count}")
    print(f"[LOADER INFO] === SMARTS文件 '{file_path}' 加载完成。共加载 {loaded_smarts_count} 个有效定义。 ===")
    return smarts_dict


def _get_functional_group_fragments(mol, smarts_dictionary, original_bonds_info, num_original_atoms):
    print("\n  [FG_FRAGMENTER] === 开始通过SMARTS匹配生成官能团片段 ===")
    all_fragments_list = []
    added_instance_signatures = set()
    if not smarts_dictionary:
        print("  [FG_FRAGMENTER INFO] SMARTS字典为空，无法进行匹配。")
        return all_fragments_list
    total_raw_matches_across_all_smarts = 0
    unique_fragments_from_this_fn = 0
    for fg_name, fg_smarts in smarts_dictionary.items():
        try:
            query_mol = Chem.MolFromSmarts(fg_smarts)
            if not query_mol:
                print(f"      [FG_FRAGMENTER WARN] SMARTS '{fg_smarts}' (名称: {fg_name}) 无效，跳过。")
                continue
            matches_atom_indices_tuples = mol.GetSubstructMatches(query_mol, uniquify=False)
            total_raw_matches_across_all_smarts += len(matches_atom_indices_tuples)
            if not matches_atom_indices_tuples:
                continue
            print(f"      [FG_FRAGMENTER DETAIL] '{fg_name}' 找到 {len(matches_atom_indices_tuples)} 个原始匹配实例。")
            for match_idx, atom_indices_tuple in enumerate(matches_atom_indices_tuples):
                print(
                    f"        [FG_FRAGMENTER SUB-DETAIL] --- 处理来自 '{fg_name}' 的匹配实例 {match_idx + 1}/{len(matches_atom_indices_tuples)} ---")
                print(f"          [FG_FRAGMENTER SUB-DETAIL] 匹配到的原子索引元组: {atom_indices_tuple}")
                current_frag_atoms = list(atom_indices_tuple)
                valid = all(isinstance(idx, int) and 0 <= idx < num_original_atoms for idx in current_frag_atoms)
                if not valid or not current_frag_atoms:
                    print(f"          [FG_FRAGMENTER WARN] 此匹配实例因无效原子索引或为空而被跳过。")
                    continue
                current_frag_atom_set = set(current_frag_atoms)
                current_frag_bonds = []
                for bond_idx, (a1, a2) in original_bonds_info.items():
                    if a1 in current_frag_atom_set and a2 in current_frag_atom_set:
                        current_frag_bonds.append(bond_idx)
                print(
                    f"          [FG_FRAGMENTER SUB-DETAIL] 推断出的内部键索引 ({len(current_frag_bonds)}个): {sorted(current_frag_bonds)}")
                current_frag_atoms.sort()
                current_frag_bonds.sort()
                instance_signature = (frozenset(current_frag_atoms), frozenset(current_frag_bonds))
                if instance_signature not in added_instance_signatures:
                    added_instance_signatures.add(instance_signature)
                    all_fragments_list.append([current_frag_atoms, current_frag_bonds])
                    unique_fragments_from_this_fn += 1
                    print(
                        f"          [FG_FRAGMENTER ACTION] 添加独特片段 (原子数 {len(current_frag_atoms)}, 键数 {len(current_frag_bonds)})。")
                else:
                    print(f"          [FG_FRAGMENTER INFO] 此匹配实例与已添加的片段重复，跳过。")
        except Exception as e:
            print(f"  [FG_FRAGMENTER ERROR] 处理SMARTS '{fg_name}' ('{fg_smarts}') 时出错: {e}")
            continue
    print(f"\n  [FG_FRAGMENTER INFO] 所有SMARTS共产生 {total_raw_matches_across_all_smarts} 个原始匹配实例。")
    print(f"  [FG_FRAGMENTER] === SMARTS片段化结束，共生成 {unique_fragments_from_this_fn} 个独特片段 ===")
    return all_fragments_list


def map_molecule_to_all_fragment_types(mol, smarts_rules_file_path):
    print(f"\n[MAIN_INTEGRATED_MAPPER] === 开始对分子进行多种方式的片段化/官能团识别 ===")
    print(f"  SMARTS官能团规则文件: {smarts_rules_file_path}")

    methods_enabled = []
    if cfg.use_brics_fragments:
        methods_enabled.append("BRICS")
    if cfg.use_murcko_fragments:
        methods_enabled.append("Murcko")
    if cfg.use_smarts_fragments:
        methods_enabled.append("SMARTS")
    
    print(f"  [CONFIG] 启用的motif提取方法: {', '.join(methods_enabled) if methods_enabled else 'None (仅whole_mol)'}")
    
    if mol is None:
        print("[MAIN_INTEGRATED_MAPPER ERROR] 输入 Mol 对象为 None")
        return None

    print("\n[MAIN_INTEGRATED_MAPPER DEBUG] 开始获取原始分子信息...")
    num_original_atoms = mol.GetNumAtoms()
    original_atom_indices_list = list(range(num_original_atoms))
    original_bonds_info_dict = {bond.GetIdx(): (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in
                                mol.GetBonds()}
    original_bond_indices_list = sorted(list(original_bonds_info_dict.keys()))
    print(
        f"[MAIN_INTEGRATED_MAPPER DEBUG] 原始分子原子数: {num_original_atoms}, 键数: {len(original_bonds_info_dict)}。")

    result_dict = {"whole_mol": [original_atom_indices_list, original_bond_indices_list]}
    added_final_fragment_signatures = set()
    fragment_counter = 1
    print(f"[MAIN_INTEGRATED_MAPPER DEBUG] 初始化最终 result_dict 和全局去重集合。")

    all_fragments_from_all_sources = [] 

    if cfg.use_brics_fragments:
        print("\n[MAIN_INTEGRATED_MAPPER] ===>>> 开始 BRICS 拆分流程 <<<===")
        brics_frags = _get_brics_fragments(mol, original_bonds_info_dict, num_original_atoms)
        if brics_frags:
            print(f"[MAIN_INTEGRATED_MAPPER INFO] BRICS 拆分得到 {len(brics_frags)} 个片段。")
            all_fragments_from_all_sources.extend(brics_frags)
        else:
            print("[MAIN_INTEGRATED_MAPPER INFO] BRICS 拆分未得到片段。")
    else:
        print("\n[MAIN_INTEGRATED_MAPPER] ===>>> BRICS 拆分已禁用（消融实验）<<<===")

    if cfg.use_murcko_fragments:
        print("\n[MAIN_INTEGRATED_MAPPER] ===>>> 开始 Murcko 骨架拆分流程 <<<===")
        murcko_frags = _get_murcko_fragments(mol, original_bonds_info_dict, num_original_atoms)
        if murcko_frags:
            print(f"[MAIN_INTEGRATED_MAPPER INFO] Murcko 拆分得到 {len(murcko_frags)} 个片段。")
            all_fragments_from_all_sources.extend(murcko_frags)
        else:
            print("[MAIN_INTEGRATED_MAPPER INFO] Murcko 拆分未得到片段。")
    else:
        print("\n[MAIN_INTEGRATED_MAPPER] ===>>> Murcko 骨架拆分已禁用（消融实验）<<<===")

    if cfg.use_smarts_fragments:
        print("\n[MAIN_INTEGRATED_MAPPER] ===>>> 开始 SMARTS 官能团识别流程 <<<===")
        smarts_dictionary = load_smarts_from_final_fg_txt(smarts_rules_file_path)
        if not smarts_dictionary:
            print("[MAIN_INTEGRATED_MAPPER WARN] 未能从SMARTS文件加载定义，跳过SMARTS官能团识别。")
        else:
            fg_frags = _get_functional_group_fragments(
                mol,
                smarts_dictionary,
                original_bonds_info_dict,
                num_original_atoms
            )
            if fg_frags:
                print(f"[MAIN_INTEGRATED_MAPPER INFO] SMARTS官能团识别得到 {len(fg_frags)} 个实例(片段)。")
                all_fragments_from_all_sources.extend(fg_frags)
            else:
                print("[MAIN_INTEGRATED_MAPPER INFO] SMARTS官能团识别未得到片段。")
    else:
        print("\n[MAIN_INTEGRATED_MAPPER] ===>>> SMARTS 官能团识别已禁用（消融实验）<<<===")

    print(
        f"\n[MAIN_INTEGRATED_MAPPER] === 开始对所有来源的 {len(all_fragments_from_all_sources)} 个候选片段进行全局去重 ===")

    unique_fragments_added_to_dict = 0
    for frag_idx, fragment_data_list in enumerate(all_fragments_from_all_sources):
        atoms_list = fragment_data_list[0]
        bonds_list = fragment_data_list[1]

        final_fragment_signature = (frozenset(atoms_list), frozenset(bonds_list))

        if final_fragment_signature not in added_final_fragment_signatures:
            added_final_fragment_signatures.add(final_fragment_signature)
            fragment_key = f"frag_{fragment_counter}"
            result_dict[fragment_key] = fragment_data_list
            fragment_counter += 1
            unique_fragments_added_to_dict += 1

    final_actual_fragment_count = len(result_dict) - 1
    print(f"\n[MAIN_INTEGRATED_MAPPER] === 所有片段化/官能团识别及合并去重流程结束 ===")
    print(f"[MAIN_INTEGRATED_MAPPER INFO] 全局去重后，共添加 {unique_fragments_added_to_dict} 个独特的片段到结果字典。")
    print(f"[MAIN_INTEGRATED_MAPPER INFO] 最终生成 {final_actual_fragment_count} 个独特的片段条目 (不含 whole_mol)。")
    print(f"[MAIN_INTEGRATED_MAPPER INFO] 使用的方法配置: {', '.join(methods_enabled) if methods_enabled else 'None'}")
    
    return result_dict