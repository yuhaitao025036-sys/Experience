"""test_dynamic_scope_go_to_definition: generated test cases."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ast_tag_db import load_jsonl_dataset_into_ast_tag_db, AstTagDB
from tag_actions.dynamic_scope_go_to_definition import dynamic_scope_go_to_definition


def _db() -> AstTagDB:
    if not hasattr(_db, "_instance"):
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
        _db._instance = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    return _db._instance


class TestDynamicScopeGoToDefinition(unittest.TestCase):

    def test_000__build_nested(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_build_nested')
        self.assertGreaterEqual(len(defs), 1)

    def test_001__get_storage_path(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_storage_path')
        self.assertGreaterEqual(len(defs), 1)

    def test_010_frame_to_str(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'frame_to_str')
        self.assertGreaterEqual(len(defs), 1)

    def test_013_ClaudeAgentOptions(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'ClaudeAgentOptions')
        self.assertGreaterEqual(len(defs), 1)

    def test_014_func(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'func')
        self.assertGreaterEqual(len(defs), 1)

    def test_015_query(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'query')
        self.assertGreaterEqual(len(defs), 1)

    def test_016__do_one(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_do_one')
        self.assertGreaterEqual(len(defs), 1)

    def test_017__flatten_nested(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_flatten_nested')
        self.assertGreaterEqual(len(defs), 1)

    def test_018__run_all(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_run_all')
        self.assertGreaterEqual(len(defs), 1)

    def test_019_coding_agent_query(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'coding_agent_query')
        self.assertGreaterEqual(len(defs), 1)

    def test_021_AsyncOpenAI(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'AsyncOpenAI')
        self.assertGreaterEqual(len(defs), 1)

    def test_022__grep_by_file_content_hint(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_grep_by_file_content_hint')
        self.assertGreaterEqual(len(defs), 1)

    def test_023_pack_dir(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'pack_dir')
        self.assertGreaterEqual(len(defs), 1)

    def test_024_raw_llm_query(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'raw_llm_query')
        self.assertGreaterEqual(len(defs), 1)

    def test_025_CodingAgentTaskHandler(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'CodingAgentTaskHandler')
        self.assertGreaterEqual(len(defs), 1)

    def test_026_RawLlmTaskHandler(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'RawLlmTaskHandler')
        self.assertGreaterEqual(len(defs), 1)

    def test_028__collect(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_collect')
        self.assertGreaterEqual(len(defs), 1)

    def test_029__is_leaf(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_is_leaf')
        self.assertGreaterEqual(len(defs), 1)

    def test_031_defaultdict(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'defaultdict')
        self.assertGreaterEqual(len(defs), 1)

    def test_033_Path(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'Path')
        self.assertGreaterEqual(len(defs), 1)

    def test_034_SoleFileBatchDataLoader(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'SoleFileBatchDataLoader')
        self.assertGreaterEqual(len(defs), 1)

    def test_035__get_all_file_paths(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_all_file_paths')
        self.assertGreaterEqual(len(defs), 1)

    def test_038_make_tensor(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'make_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_041_read_storage(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        self.assertGreaterEqual(len(defs), 1)

    def test_042_run_test(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        self.assertGreaterEqual(len(defs), 1)

    def test_044_storage_path(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'storage_path')
        self.assertGreaterEqual(len(defs), 1)

    def test_045_AgentTask(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'AgentTask')
        self.assertGreaterEqual(len(defs), 1)

    def test_046_TaskHandler(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'TaskHandler')
        self.assertGreaterEqual(len(defs), 1)

    def test_047__build_nested_result(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_build_nested_result')
        self.assertGreaterEqual(len(defs), 1)

    def test_048__copy_back_to_storage_view(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_copy_back_to_storage_view')
        self.assertGreaterEqual(len(defs), 1)

    def test_049__scalar_slice_indices(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_scalar_slice_indices')
        self.assertGreaterEqual(len(defs), 1)

    def test_050_coding_agent(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'coding_agent')
        self.assertGreaterEqual(len(defs), 1)

    def test_051_dump_view(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'dump_view')
        self.assertGreaterEqual(len(defs), 1)

    def test_052_read_output(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_output')
        self.assertGreaterEqual(len(defs), 1)

    def test_054_slice_tensor(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_055_slice_view(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_view')
        self.assertGreaterEqual(len(defs), 1)

    def test_056_todo_tensor_like(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'todo_tensor_like')
        self.assertGreaterEqual(len(defs), 1)

    def test_058_assign_tensor(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'assign_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_059_fork_tensor_backward(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'fork_tensor_backward')
        self.assertGreaterEqual(len(defs), 1)

    def test_060_fork_tensor_forward(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'fork_tensor_forward')
        self.assertGreaterEqual(len(defs), 1)

    def test_061_get_diff_tensor(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_diff_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_065_get_causal_attention_mask(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_causal_attention_mask')
        self.assertGreaterEqual(len(defs), 1)

    def test_066__get_diff(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_diff')
        self.assertGreaterEqual(len(defs), 1)

    def test_067__read_storage(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_read_storage')
        self.assertGreaterEqual(len(defs), 1)

    def test_068__unflatten(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_unflatten')
        self.assertGreaterEqual(len(defs), 1)

    def test_069_get_edit_distance_ratio_backward_impl(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_edit_distance_ratio_backward_impl')
        self.assertGreaterEqual(len(defs), 1)

    def test_070_get_edit_distance_ratio_impl(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_edit_distance_ratio_impl')
        self.assertGreaterEqual(len(defs), 1)

    def test_073_get_query_tensor(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_query_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_075_merge(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'merge')
        self.assertGreaterEqual(len(defs), 1)

    def test_076_merge_backward(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'merge_backward')
        self.assertGreaterEqual(len(defs), 1)

    def test_077_merge_forward(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'merge_forward')
        self.assertGreaterEqual(len(defs), 1)

    def test_078_read_out(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_out')
        self.assertGreaterEqual(len(defs), 1)

    def test_079_st_patched(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'st_patched')
        self.assertGreaterEqual(len(defs), 1)

    def test_080__write_storage(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_write_storage')
        self.assertGreaterEqual(len(defs), 1)

    def test_081_make_none_tensor(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'make_none_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_082__extract_coordinates(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_extract_coordinates')
        self.assertGreaterEqual(len(defs), 1)

    def test_083__filter_last_coordinate_eq_zero(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_filter_last_coordinate_eq_zero')
        self.assertGreaterEqual(len(defs), 1)

    def test_084__unzip_to_tensor_list(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_unzip_to_tensor_list')
        self.assertGreaterEqual(len(defs), 1)

    def test_086_select_qkv_indexes(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'select_qkv_indexes')
        self.assertGreaterEqual(len(defs), 1)

    def test_088_none_tensor_like(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'none_tensor_like')
        self.assertGreaterEqual(len(defs), 1)

    def test_089_slice_and_concat_attention_forward(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_and_concat_attention_forward')
        self.assertGreaterEqual(len(defs), 1)

    def test_090_slice_attention(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_attention')
        self.assertGreaterEqual(len(defs), 1)

    def test_091_slice_attention_backward(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_attention_backward')
        self.assertGreaterEqual(len(defs), 1)

    def test_092_slice_attention_forward(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_attention_forward')
        self.assertGreaterEqual(len(defs), 1)

    def test_093_slice_attention_backward_grad_input(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_attention_backward_grad_input')
        self.assertGreaterEqual(len(defs), 1)

    def test_094__get_raw_storage_path(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_raw_storage_path')
        self.assertGreaterEqual(len(defs), 1)

    def test_095__build_per_dim(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_build_per_dim')
        self.assertGreaterEqual(len(defs), 1)

    def test_097__resolve_grad_output(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_resolve_grad_output')
        self.assertGreaterEqual(len(defs), 1)

    def test_098__restore_st_attrs(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_restore_st_attrs')
        self.assertGreaterEqual(len(defs), 1)

    def test_099__save_st_attrs(self):
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_save_st_attrs')
        self.assertGreaterEqual(len(defs), 1)


if __name__ == "__main__":
    unittest.main()
