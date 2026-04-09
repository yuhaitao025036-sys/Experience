"""test_dynamic_scope_find_all_references: generated test cases."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ast_tag_db import load_jsonl_dataset_into_ast_tag_db, AstTagDB
from tag_actions.dynamic_scope_find_all_references import dynamic_scope_find_all_references


def _db() -> AstTagDB:
    if not hasattr(_db, "_instance"):
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
        _db._instance = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    return _db._instance


class TestDynamicScopeFindAllReferences(unittest.TestCase):

    def test_000__build_nested(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_build_nested')
        self.assertGreaterEqual(len(refs), 1)

    def test_001__get_storage_path(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_storage_path')
        self.assertGreaterEqual(len(refs), 1)

    def test_002_get_nested_list_file_pathes(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'get_nested_list_file_pathes')
        self.assertGreaterEqual(len(refs), 1)

    def test_003_pack_dir(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'pack_dir')
        self.assertGreaterEqual(len(refs), 1)

    def test_004_frame_to_str(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'frame_to_str')
        self.assertGreaterEqual(len(refs), 1)

    def test_005_pack(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'pack')
        self.assertGreaterEqual(len(refs), 1)

    def test_006_unpack(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'unpack')
        self.assertGreaterEqual(len(refs), 1)

    def test_009__flatten_nested(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_flatten_nested')
        self.assertGreaterEqual(len(refs), 1)

    def test_010__grep_by_file_content_hint(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_grep_by_file_content_hint')
        self.assertGreaterEqual(len(refs), 1)

    def test_011__collect(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_collect')
        self.assertGreaterEqual(len(refs), 1)

    def test_012__is_leaf(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_is_leaf')
        self.assertGreaterEqual(len(refs), 1)

    def test_013_convert_nested_list_coordinates_to_pairs_coordinates(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'convert_nested_list_coordinates_to_pairs_coordinates')
        self.assertGreaterEqual(len(refs), 1)

    def test_015_transpose_pairs_coordinates(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'transpose_pairs_coordinates')
        self.assertGreaterEqual(len(refs), 1)

    def test_016___init__(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '__init__')
        self.assertGreaterEqual(len(refs), 1)

    def test_019__get_all_file_paths(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_all_file_paths')
        self.assertGreaterEqual(len(refs), 1)

    def test_020_read_storage(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        self.assertGreaterEqual(len(refs), 1)

    def test_021_run_test(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertGreaterEqual(len(refs), 1)

    def test_022_storage_path(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'storage_path')
        self.assertGreaterEqual(len(refs), 1)

    def test_023__build_nested_result(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_build_nested_result')
        self.assertGreaterEqual(len(refs), 1)

    def test_024__copy_back_to_storage_view(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_copy_back_to_storage_view')
        self.assertGreaterEqual(len(refs), 1)

    def test_025__scalar_slice_indices(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_scalar_slice_indices')
        self.assertGreaterEqual(len(refs), 1)

    def test_026_coding_agent(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'coding_agent')
        self.assertGreaterEqual(len(refs), 1)

    def test_027_custom_prompt(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'custom_prompt')
        self.assertGreaterEqual(len(refs), 1)

    def test_028_default_prompt_for_output(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_output')
        self.assertGreaterEqual(len(refs), 1)

    def test_029_read_output(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_output')
        self.assertGreaterEqual(len(refs), 1)

    def test_030_backward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'backward')
        self.assertGreaterEqual(len(refs), 1)

    def test_031_default_prompt_for_fork_grad_input(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_fork_grad_input')
        self.assertGreaterEqual(len(refs), 1)

    def test_032_fork_tensor_backward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'fork_tensor_backward')
        self.assertGreaterEqual(len(refs), 1)

    def test_033_fork_tensor_forward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'fork_tensor_forward')
        self.assertGreaterEqual(len(refs), 1)

    def test_034_forward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'forward')
        self.assertGreaterEqual(len(refs), 1)

    def test_035_get_causal_attention_mask(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'get_causal_attention_mask')
        self.assertGreaterEqual(len(refs), 1)

    def test_036__get_diff(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_diff')
        self.assertGreaterEqual(len(refs), 1)

    def test_037__read_storage(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_read_storage')
        self.assertGreaterEqual(len(refs), 1)

    def test_038__unflatten(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_unflatten')
        self.assertGreaterEqual(len(refs), 1)

    def test_039_get_edit_distance_ratio_backward_impl(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'get_edit_distance_ratio_backward_impl')
        self.assertGreaterEqual(len(refs), 1)

    def test_040_get_edit_distance_ratio_impl(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'get_edit_distance_ratio_impl')
        self.assertGreaterEqual(len(refs), 1)

    def test_041_default_prompt_for_query(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_query')
        self.assertGreaterEqual(len(refs), 1)

    def test_042_get_query_tensor(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'get_query_tensor')
        self.assertGreaterEqual(len(refs), 1)

    def test_043_my_prompt(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'my_prompt')
        self.assertGreaterEqual(len(refs), 1)

    def test_044_merge_backward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'merge_backward')
        self.assertGreaterEqual(len(refs), 1)

    def test_045_read_out(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_out')
        self.assertGreaterEqual(len(refs), 1)

    def test_046__write_storage(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_write_storage')
        self.assertGreaterEqual(len(refs), 1)

    def test_047_merge_forward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'merge_forward')
        self.assertGreaterEqual(len(refs), 1)

    def test_048__extract_coordinates(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_extract_coordinates')
        self.assertGreaterEqual(len(refs), 1)

    def test_049__filter_last_coordinate_eq_zero(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_filter_last_coordinate_eq_zero')
        self.assertGreaterEqual(len(refs), 1)

    def test_050__unzip_to_tensor_list(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_unzip_to_tensor_list')
        self.assertGreaterEqual(len(refs), 1)

    def test_051_default_retrieval_method(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_retrieval_method')
        self.assertGreaterEqual(len(refs), 1)

    def test_052_exact_match(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'exact_match')
        self.assertGreaterEqual(len(refs), 1)

    def test_053_select_qkv_indexes(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'select_qkv_indexes')
        self.assertGreaterEqual(len(refs), 1)

    def test_054_slice_and_concat_attention_forward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_and_concat_attention_forward')
        self.assertGreaterEqual(len(refs), 1)

    def test_055_default_prompt_for_grad_input(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_grad_input')
        self.assertGreaterEqual(len(refs), 1)

    def test_056_slice_attention_backward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_attention_backward')
        self.assertGreaterEqual(len(refs), 1)

    def test_057_slice_attention_backward_grad_input(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_attention_backward_grad_input')
        self.assertGreaterEqual(len(refs), 1)

    def test_058__get_raw_storage_path(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_raw_storage_path')
        self.assertGreaterEqual(len(refs), 1)

    def test_059_slice_attention_forward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_attention_forward')
        self.assertGreaterEqual(len(refs), 1)

    def test_060_slice_tensor(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_tensor')
        self.assertGreaterEqual(len(refs), 1)

    def test_061__build_per_dim(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_build_per_dim')
        self.assertGreaterEqual(len(refs), 1)

    def test_062__coords_to_flat(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_coords_to_flat')
        self.assertGreaterEqual(len(refs), 1)

    def test_063__resolve_grad_output(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_resolve_grad_output')
        self.assertGreaterEqual(len(refs), 1)

    def test_064__restore_st_attrs(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_restore_st_attrs')
        self.assertGreaterEqual(len(refs), 1)

    def test_065__save_st_attrs(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_save_st_attrs')
        self.assertGreaterEqual(len(refs), 1)

    def test_066_slice_backward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_backward')
        self.assertGreaterEqual(len(refs), 1)

    def test_067_slice_view(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_view')
        self.assertGreaterEqual(len(refs), 1)

    def test_068_st_attention(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_attention')
        self.assertGreaterEqual(len(refs), 1)

    def test_069_copy_impl(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'copy_impl')
        self.assertGreaterEqual(len(refs), 1)

    def test_070__detect_input_content_type(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_detect_input_content_type')
        self.assertGreaterEqual(len(refs), 1)

    def test_071__flatten_nested_indexes(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_flatten_nested_indexes')
        self.assertGreaterEqual(len(refs), 1)

    def test_072__force_todo_nested(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_force_todo_nested')
        self.assertGreaterEqual(len(refs), 1)

    def test_073__merge_and_shuffle_and_select_prefix_topk(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_merge_and_shuffle_and_select_prefix_topk')
        self.assertGreaterEqual(len(refs), 1)

    def test_074__pad_indexes_to_topk_with_none_experience_indexes(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_pad_indexes_to_topk_with_none_experience_indexes')
        self.assertGreaterEqual(len(refs), 1)

    def test_076__replace_last_tensor_with_full_slice(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_replace_last_tensor_with_full_slice')
        self.assertGreaterEqual(len(refs), 1)

    def test_077__replace_last_tensor_with_slice(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_replace_last_tensor_with_slice')
        self.assertGreaterEqual(len(refs), 1)

    def test_078__select_random_indexes_with_none_experience_indexes(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_select_random_indexes_with_none_experience_indexes')
        self.assertGreaterEqual(len(refs), 1)

    def test_079_default_prompt_for_grad_exp_key(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_grad_exp_key')
        self.assertGreaterEqual(len(refs), 1)

    def test_080_default_prompt_for_grad_exp_value(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_grad_exp_value')
        self.assertGreaterEqual(len(refs), 1)

    def test_081_default_prompt_for_grad_input_frame(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_grad_input_frame')
        self.assertGreaterEqual(len(refs), 1)

    def test_082_st_moe_backward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_moe_backward')
        self.assertGreaterEqual(len(refs), 1)

    def test_083_st_moe_backward_grad_experience(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_moe_backward_grad_experience')
        self.assertGreaterEqual(len(refs), 1)

    def test_084_st_moe_backward_grad_input(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_moe_backward_grad_input')
        self.assertGreaterEqual(len(refs), 1)

    def test_085__read_file_content(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_read_file_content')
        self.assertGreaterEqual(len(refs), 1)

    def test_086_st_moe_forward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_moe_forward')
        self.assertGreaterEqual(len(refs), 1)

    def test_087_st_stack(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_stack')
        self.assertGreaterEqual(len(refs), 1)

    def test_088_st_stack_backward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_stack_backward')
        self.assertGreaterEqual(len(refs), 1)

    def test_089_st_stack_forward(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_stack_forward')
        self.assertGreaterEqual(len(refs), 1)

    def test_090__get_store(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_store')
        self.assertGreaterEqual(len(refs), 1)

    def test_092_pop(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'pop')
        self.assertGreaterEqual(len(refs), 1)

    def test_093_register(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'register')
        self.assertGreaterEqual(len(refs), 1)

    def test_094_with_dense_view(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'with_dense_view')
        self.assertGreaterEqual(len(refs), 1)

    def test_095_zero_middle(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'zero_middle')
        self.assertGreaterEqual(len(refs), 1)

    def test_096_parameters(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'parameters')
        self.assertGreaterEqual(len(refs), 1)

    def test_097_zero_first(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'zero_first')
        self.assertGreaterEqual(len(refs), 1)

    def test_098__flatten_nested_paths(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_flatten_nested_paths')
        self.assertGreaterEqual(len(refs), 1)

    def test_099__get_nonzero_points(self):
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_nonzero_points')
        self.assertGreaterEqual(len(refs), 1)


if __name__ == "__main__":
    unittest.main()
