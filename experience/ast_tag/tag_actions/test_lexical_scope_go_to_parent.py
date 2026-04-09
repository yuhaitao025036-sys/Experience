"""test_lexical_scope_go_to_parent: generated test cases."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ast_tag_db import load_jsonl_dataset_into_ast_tag_db, AstTagDB
from relation_tag_classification import LEXICAL_RELATION_TAGS
from tag_actions.lexical_scope_go_to_parent import lexical_scope_go_to_parent


def _db() -> AstTagDB:
    if not hasattr(_db, "_instance"):
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
        _db._instance = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    return _db._instance


class TestLexicalScopeGoToParent(unittest.TestCase):

    def test_000_fs_util_get_nested_list_file_pathes_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'fs_util/get_nested_list_file_pathes.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_001_fs_util_get_nested_list_file_pathes_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'fs_util/get_nested_list_file_pathes.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_002_fs_util_pack_dir_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'fs_util/pack_dir.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_003_fs_util_pack_dir_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'fs_util/pack_dir.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_004_fs_util_text_merger_jsonl__AnnAssign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'fs_util/text_merger.jsonl', '$AnnAssign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_2')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_005_fs_util_text_merger_jsonl__AnnAssign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'fs_util/text_merger.jsonl', '$AnnAssign_1')
        self.assertEqual(parent.owner_tag, '$For_0')
        self.assertEqual(parent.relation_tag, 'For.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_006_llm_client_agent_task_jsonl__AnnAssign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/agent_task.jsonl', '$AnnAssign_0')
        self.assertEqual(parent.owner_tag, '$ClassDef_0')
        self.assertEqual(parent.relation_tag, 'ClassDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_007_llm_client_agent_task_jsonl__AnnAssign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/agent_task.jsonl', '$AnnAssign_1')
        self.assertEqual(parent.owner_tag, '$ClassDef_0')
        self.assertEqual(parent.relation_tag, 'ClassDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_008_llm_client_coding_agent_query_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_query.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$If_0')
        self.assertEqual(parent.relation_tag, 'If.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_009_llm_client_coding_agent_query_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_query.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$AsyncFunctionDef_0')
        self.assertEqual(parent.relation_tag, 'AsyncFunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_010_llm_client_coding_agent_task_handler_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_task_handler.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_011_llm_client_coding_agent_task_handler_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_task_handler.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_012_llm_client_raw_llm_query_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_query.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_013_llm_client_raw_llm_query_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_query.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$AsyncFunctionDef_0')
        self.assertEqual(parent.relation_tag, 'AsyncFunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_014_llm_client_raw_llm_task_handler_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_task_handler.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_015_llm_client_raw_llm_task_handler_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_task_handler.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_016_llm_client_task_handler_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/task_handler.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_017_llm_client_task_handler_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/task_handler.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_018_sparse_util_convert_nested_list_coordinates_to_pairs_coordinates_jsonl__AnnAssign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$AnnAssign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_2')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_019_sparse_util_convert_nested_list_coordinates_to_pairs_coordinates_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_020_sparse_util_group_random_select_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'sparse_util/group_random_select.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_021_sparse_util_group_random_select_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'sparse_util/group_random_select.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_022_sparse_util_transpose_pairs_coordinates_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'sparse_util/transpose_pairs_coordinates.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_023_sparse_util_transpose_pairs_coordinates_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'sparse_util/transpose_pairs_coordinates.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_024_symbolic_tensor_data_loader_sole_file_batch_data_loader_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_025_symbolic_tensor_data_loader_sole_file_batch_data_loader_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_026_symbolic_tensor_function_coding_agent_jsonl__AnnAssign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/coding_agent.jsonl', '$AnnAssign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_4')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_027_symbolic_tensor_function_coding_agent_jsonl__AnnAssign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/coding_agent.jsonl', '$AnnAssign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_4')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_028_symbolic_tensor_function_fork_tensor_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/fork_tensor.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_029_symbolic_tensor_function_fork_tensor_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/fork_tensor.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_030_symbolic_tensor_function_get_causal_attention_mask_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_031_symbolic_tensor_function_get_causal_attention_mask_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_032_symbolic_tensor_function_get_edit_distance_ratio_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_033_symbolic_tensor_function_get_edit_distance_ratio_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_034_symbolic_tensor_function_get_query_tensor_jsonl__Assert_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_query_tensor.jsonl', '$Assert_0')
        self.assertEqual(parent.owner_tag, '$With_3')
        self.assertEqual(parent.relation_tag, 'With.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_035_symbolic_tensor_function_get_query_tensor_jsonl__Assert_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_query_tensor.jsonl', '$Assert_1')
        self.assertEqual(parent.owner_tag, '$If_4')
        self.assertEqual(parent.relation_tag, 'If.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_036_symbolic_tensor_function_merge_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_037_symbolic_tensor_function_merge_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_038_symbolic_tensor_function_merge_backward_jsonl__Assert_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', '$Assert_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_2')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_039_symbolic_tensor_function_merge_backward_jsonl__Assert_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', '$Assert_1')
        self.assertEqual(parent.owner_tag, '$For_0')
        self.assertEqual(parent.relation_tag, 'For.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_040_symbolic_tensor_function_merge_forward_jsonl__Assert_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_forward.jsonl', '$Assert_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_3')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_041_symbolic_tensor_function_merge_forward_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_forward.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_042_symbolic_tensor_function_select_qkv_indexes_jsonl__AnnAssign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$AnnAssign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_4')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_043_symbolic_tensor_function_select_qkv_indexes_jsonl__AnnAssign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$AnnAssign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_4')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_044_symbolic_tensor_function_slice_and_concat_attention_forward_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_045_symbolic_tensor_function_slice_and_concat_attention_forward_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_046_symbolic_tensor_function_slice_attention_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_047_symbolic_tensor_function_slice_attention_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_048_symbolic_tensor_function_slice_attention_backward_jsonl__AnnAssign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$AnnAssign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_049_symbolic_tensor_function_slice_attention_backward_jsonl__AnnAssign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$AnnAssign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_050_symbolic_tensor_function_slice_attention_forward_jsonl__Assert_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$Assert_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_051_symbolic_tensor_function_slice_attention_forward_jsonl__Assert_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$Assert_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_052_symbolic_tensor_function_slice_tensor_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_tensor.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_053_symbolic_tensor_function_slice_tensor_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_tensor.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_054_symbolic_tensor_function_slice_view_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_view.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_055_symbolic_tensor_function_slice_view_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_view.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$For_0')
        self.assertEqual(parent.relation_tag, 'For.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_056_symbolic_tensor_function_st_attention_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_attention.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_057_symbolic_tensor_function_st_attention_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_attention.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_2')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_058_symbolic_tensor_function_st_copy_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_copy.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_2')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_059_symbolic_tensor_function_st_copy_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_copy.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_2')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_060_symbolic_tensor_function_st_moe_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_061_symbolic_tensor_function_st_moe_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_062_symbolic_tensor_function_st_moe_backward_jsonl__AnnAssign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$AnnAssign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_6')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_063_symbolic_tensor_function_st_moe_backward_jsonl__AnnAssign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$AnnAssign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_7')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_064_symbolic_tensor_function_st_moe_forward_jsonl__AnnAssign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_forward.jsonl', '$AnnAssign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_065_symbolic_tensor_function_st_moe_forward_jsonl__AnnAssign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_forward.jsonl', '$AnnAssign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_6')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_066_symbolic_tensor_function_st_stack_jsonl__AnnAssign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', '$AnnAssign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_5')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_067_symbolic_tensor_function_st_stack_jsonl__Assert_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', '$Assert_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_4')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_068_symbolic_tensor_function_symbolic_grad_registry_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_069_symbolic_tensor_function_symbolic_grad_registry_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$If_0')
        self.assertEqual(parent.relation_tag, 'If.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_070_symbolic_tensor_function_with_dense_view_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/with_dense_view.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_071_symbolic_tensor_function_with_dense_view_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/with_dense_view.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_072_symbolic_tensor_module_st_moe_jsonl__AnnAssign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/st_moe.jsonl', '$AnnAssign_0')
        self.assertEqual(parent.owner_tag, '$ClassDef_0')
        self.assertEqual(parent.relation_tag, 'ClassDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_073_symbolic_tensor_module_st_moe_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/st_moe.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_074_symbolic_tensor_module_with_dense_view_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/with_dense_view.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_075_symbolic_tensor_module_with_dense_view_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/with_dense_view.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_3')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_076_symbolic_tensor_optimizer_st_sgd_jsonl__AnnAssign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$AnnAssign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_2')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_077_symbolic_tensor_optimizer_st_sgd_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_078_symbolic_tensor_tensor_util_assign_tensor_jsonl__Assert_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$Assert_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_079_symbolic_tensor_tensor_util_assign_tensor_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_080_symbolic_tensor_tensor_util_assign_view_jsonl__Assert_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$Assert_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_081_symbolic_tensor_tensor_util_assign_view_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_082_symbolic_tensor_tensor_util_dense_to_sparse_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_083_symbolic_tensor_tensor_util_dense_to_sparse_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_084_symbolic_tensor_tensor_util_dump_tensor_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_085_symbolic_tensor_tensor_util_dump_tensor_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_086_symbolic_tensor_tensor_util_dump_view_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_087_symbolic_tensor_tensor_util_dump_view_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_3')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_088_symbolic_tensor_tensor_util_empty_tensor_like_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_089_symbolic_tensor_tensor_util_empty_tensor_like_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_090_symbolic_tensor_tensor_util_get_diff_tensor_jsonl__Assert_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$Assert_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_2')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_091_symbolic_tensor_tensor_util_get_diff_tensor_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_092_symbolic_tensor_tensor_util_load_tensor_jsonl__Assert_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$Assert_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_093_symbolic_tensor_tensor_util_load_tensor_jsonl__Assert_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$Assert_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_094_symbolic_tensor_tensor_util_make_none_tensor_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_095_symbolic_tensor_tensor_util_make_none_tensor_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_096_symbolic_tensor_tensor_util_make_tensor_jsonl__Assert_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$Assert_0')
        self.assertEqual(parent.owner_tag, '$If_2')
        self.assertEqual(parent.relation_tag, 'If.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_097_symbolic_tensor_tensor_util_make_tensor_jsonl__Assert_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$Assert_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_2')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_098_symbolic_tensor_tensor_util_none_tensor_like_jsonl__Assign_0(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)

    def test_099_symbolic_tensor_tensor_util_none_tensor_like_jsonl__Assign_1(self):
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$Assign_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef.body')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)


if __name__ == "__main__":
    unittest.main()
