"""test_lexical_scope_expand_children: generated test cases."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ast_tag_db import load_jsonl_dataset_into_ast_tag_db, AstTagDB
from tag_actions.lexical_scope_expand_children import lexical_scope_expand_children


def _db() -> AstTagDB:
    if not hasattr(_db, "_instance"):
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
        _db._instance = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    return _db._instance


class TestLexicalScopeExpandChildren(unittest.TestCase):

    def test_000_fs_util_get_nested_list_file_pathes_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/get_nested_list_file_pathes.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$AugAssign_0', child_tags)

    def test_001_fs_util_get_nested_list_file_pathes_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/get_nested_list_file_pathes.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_002_fs_util_pack_dir_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/pack_dir.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$For_1', child_tags)

    def test_003_fs_util_pack_dir_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/pack_dir.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_2', child_tags)

    def test_004_fs_util_text_merger_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/text_merger.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_4', child_tags)

    def test_005_fs_util_text_merger_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/text_merger.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$If_2', child_tags)

    def test_006_llm_client_agent_task_jsonl__module_(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/agent_task.jsonl', '<module>')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$ImportFrom_0', child_tags)

    def test_007_llm_client_coding_agent_query_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_query.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_2', child_tags)

    def test_008_llm_client_coding_agent_query_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_query.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$If_3', child_tags)

    def test_009_llm_client_coding_agent_task_handler_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_task_handler.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Expr_1', child_tags)

    def test_010_llm_client_coding_agent_task_handler_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_task_handler.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_011_llm_client_raw_llm_query_jsonl__module_(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_query.jsonl', '<module>')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Import_0', child_tags)

    def test_012_llm_client_raw_llm_task_handler_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Expr_1', child_tags)

    def test_013_llm_client_raw_llm_task_handler_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$For_2', child_tags)

    def test_014_llm_client_task_handler_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/task_handler.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_015_llm_client_task_handler_jsonl__If_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/task_handler.jsonl', '$If_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Expr_0', child_tags)

    def test_016_sparse_util_convert_nested_list_coordinates_to_pairs_coordinates_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Expr_3', child_tags)

    def test_017_sparse_util_convert_nested_list_coordinates_to_pairs_coordinates_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_018_sparse_util_group_random_select_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'sparse_util/group_random_select.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_019_sparse_util_group_random_select_jsonl__module_(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'sparse_util/group_random_select.jsonl', '<module>')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Import_0', child_tags)

    def test_020_sparse_util_transpose_pairs_coordinates_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'sparse_util/transpose_pairs_coordinates.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_2', child_tags)

    def test_021_sparse_util_transpose_pairs_coordinates_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'sparse_util/transpose_pairs_coordinates.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$If_1', child_tags)

    def test_022_symbolic_tensor_data_loader_sole_file_batch_data_loader_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$For_1', child_tags)

    def test_023_symbolic_tensor_data_loader_sole_file_batch_data_loader_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$If_0', child_tags)

    def test_024_symbolic_tensor_function_coding_agent_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/coding_agent.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$AugAssign_0', child_tags)

    def test_025_symbolic_tensor_function_coding_agent_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/coding_agent.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_3', child_tags)

    def test_026_symbolic_tensor_function_fork_tensor_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/fork_tensor.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Expr_4', child_tags)

    def test_027_symbolic_tensor_function_fork_tensor_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/fork_tensor.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_7', child_tags)

    def test_028_symbolic_tensor_function_get_causal_attention_mask_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_029_symbolic_tensor_function_get_causal_attention_mask_jsonl__FunctionDef_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', '$FunctionDef_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_1', child_tags)

    def test_030_symbolic_tensor_function_get_edit_distance_ratio_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_7', child_tags)

    def test_031_symbolic_tensor_function_get_edit_distance_ratio_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_14', child_tags)

    def test_032_symbolic_tensor_function_get_query_tensor_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_query_tensor.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_1', child_tags)

    def test_033_symbolic_tensor_function_get_query_tensor_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_query_tensor.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$If_3', child_tags)

    def test_034_symbolic_tensor_function_merge_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_2', child_tags)

    def test_035_symbolic_tensor_function_merge_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$If_0', child_tags)

    def test_036_symbolic_tensor_function_merge_backward_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_12', child_tags)

    def test_037_symbolic_tensor_function_merge_backward_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_23', child_tags)

    def test_038_symbolic_tensor_function_merge_forward_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_forward.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_15', child_tags)

    def test_039_symbolic_tensor_function_merge_forward_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_forward.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_17', child_tags)

    def test_040_symbolic_tensor_function_select_qkv_indexes_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$For_1', child_tags)

    def test_041_symbolic_tensor_function_select_qkv_indexes_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_5', child_tags)

    def test_042_symbolic_tensor_function_slice_and_concat_attention_forward_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_9', child_tags)

    def test_043_symbolic_tensor_function_slice_and_concat_attention_forward_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Expr_5', child_tags)

    def test_044_symbolic_tensor_function_slice_attention_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_2', child_tags)

    def test_045_symbolic_tensor_function_slice_attention_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$If_0', child_tags)

    def test_046_symbolic_tensor_function_slice_attention_backward_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_4', child_tags)

    def test_047_symbolic_tensor_function_slice_attention_backward_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_17', child_tags)

    def test_048_symbolic_tensor_function_slice_attention_forward_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_7', child_tags)

    def test_049_symbolic_tensor_function_slice_attention_forward_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_9', child_tags)

    def test_050_symbolic_tensor_function_slice_tensor_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_tensor.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_051_symbolic_tensor_function_slice_tensor_jsonl__FunctionDef_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_tensor.jsonl', '$FunctionDef_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_1', child_tags)

    def test_052_symbolic_tensor_function_slice_view_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_view.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_1', child_tags)

    def test_053_symbolic_tensor_function_slice_view_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_view.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$If_0', child_tags)

    def test_054_symbolic_tensor_function_st_attention_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_attention.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_35', child_tags)

    def test_055_symbolic_tensor_function_st_attention_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_attention.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_52', child_tags)

    def test_056_symbolic_tensor_function_st_copy_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_copy.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Expr_17', child_tags)

    def test_057_symbolic_tensor_function_st_copy_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_copy.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_058_symbolic_tensor_function_st_moe_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_2', child_tags)

    def test_059_symbolic_tensor_function_st_moe_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$If_0', child_tags)

    def test_060_symbolic_tensor_function_st_moe_backward_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Expr_13', child_tags)

    def test_061_symbolic_tensor_function_st_moe_backward_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_11', child_tags)

    def test_062_symbolic_tensor_function_st_moe_forward_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_forward.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$AugAssign_0', child_tags)

    def test_063_symbolic_tensor_function_st_moe_forward_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_forward.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_5', child_tags)

    def test_064_symbolic_tensor_function_st_stack_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assert_1', child_tags)

    def test_065_symbolic_tensor_function_st_stack_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_13', child_tags)

    def test_066_symbolic_tensor_function_symbolic_grad_registry_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_067_symbolic_tensor_function_symbolic_grad_registry_jsonl__FunctionDef_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$FunctionDef_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_1', child_tags)

    def test_068_symbolic_tensor_function_with_dense_view_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/with_dense_view.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_12', child_tags)

    def test_069_symbolic_tensor_function_with_dense_view_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/with_dense_view.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$If_0', child_tags)

    def test_070_symbolic_tensor_module_st_moe_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$If_1', child_tags)

    def test_071_symbolic_tensor_module_st_moe_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_072_symbolic_tensor_module_with_dense_view_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/with_dense_view.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_073_symbolic_tensor_module_with_dense_view_jsonl__FunctionDef_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/with_dense_view.jsonl', '$FunctionDef_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_1', child_tags)

    def test_074_symbolic_tensor_optimizer_st_sgd_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_4', child_tags)

    def test_075_symbolic_tensor_optimizer_st_sgd_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Expr_7', child_tags)

    def test_076_symbolic_tensor_tensor_util_assign_tensor_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_5', child_tags)

    def test_077_symbolic_tensor_tensor_util_assign_tensor_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_078_symbolic_tensor_tensor_util_assign_view_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_4', child_tags)

    def test_079_symbolic_tensor_tensor_util_assign_view_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_080_symbolic_tensor_tensor_util_dense_to_sparse_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_081_symbolic_tensor_tensor_util_dense_to_sparse_jsonl__FunctionDef_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$FunctionDef_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_1', child_tags)

    def test_082_symbolic_tensor_tensor_util_dump_tensor_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_2', child_tags)

    def test_083_symbolic_tensor_tensor_util_dump_tensor_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_10', child_tags)

    def test_084_symbolic_tensor_tensor_util_dump_view_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_2', child_tags)

    def test_085_symbolic_tensor_tensor_util_dump_view_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_11', child_tags)

    def test_086_symbolic_tensor_tensor_util_empty_tensor_like_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$With_1', child_tags)

    def test_087_symbolic_tensor_tensor_util_empty_tensor_like_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$With_3', child_tags)

    def test_088_symbolic_tensor_tensor_util_get_diff_tensor_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_8', child_tags)

    def test_089_symbolic_tensor_tensor_util_get_diff_tensor_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$AugAssign_0', child_tags)

    def test_090_symbolic_tensor_tensor_util_load_tensor_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_2', child_tags)

    def test_091_symbolic_tensor_tensor_util_load_tensor_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Assign_6', child_tags)

    def test_092_symbolic_tensor_tensor_util_make_none_tensor_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_093_symbolic_tensor_tensor_util_make_none_tensor_jsonl__FunctionDef_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$FunctionDef_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_1', child_tags)

    def test_094_symbolic_tensor_tensor_util_make_tensor_jsonl__For_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$For_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Expr_3', child_tags)

    def test_095_symbolic_tensor_tensor_util_make_tensor_jsonl__For_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$For_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$Expr_5', child_tags)

    def test_096_symbolic_tensor_tensor_util_none_tensor_like_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_097_symbolic_tensor_tensor_util_none_tensor_like_jsonl__FunctionDef_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$FunctionDef_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_1', child_tags)

    def test_098_symbolic_tensor_tensor_util_pack_tensor_jsonl__FunctionDef_0(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$FunctionDef_0')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_0', child_tags)

    def test_099_symbolic_tensor_tensor_util_pack_tensor_jsonl__FunctionDef_1(self):
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$FunctionDef_1')
        self.assertGreaterEqual(len(children), 1)
        child_tags = [c.member_tag for c in children]
        self.assertIn('$arguments_1', child_tags)


if __name__ == "__main__":
    unittest.main()
