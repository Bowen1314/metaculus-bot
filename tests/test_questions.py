"""Question parsing for every type, from API-shaped JSON, through the official library."""

from __future__ import annotations

from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.data_models.questions import DateQuestion, DiscreteQuestion

from tests.conftest import log_scaled_post, parse_posts


def test_every_type_is_present_and_parsed(by_type):
    assert set(by_type) == {"binary", "multiple_choice", "numeric", "discrete", "date"}
    assert isinstance(by_type["binary"], BinaryQuestion)
    assert isinstance(by_type["multiple_choice"], MultipleChoiceQuestion)
    assert isinstance(by_type["numeric"], NumericQuestion)
    assert isinstance(by_type["discrete"], DiscreteQuestion)
    assert isinstance(by_type["date"], DateQuestion)


def test_ids_text_and_times(open_questions):
    for question in open_questions:
        assert isinstance(question.id_of_question, int)
        assert isinstance(question.id_of_post, int)
        assert question.question_text
        assert question.resolution_criteria
        assert question.close_time is not None and question.close_time.tzinfo is not None
        assert question.already_forecasted is False
        assert question.page_url.endswith(str(question.id_of_post))


def test_multiple_choice_options(by_type):
    assert by_type["multiple_choice"].options == ["Option A", "Option B", "Option C"]


def test_numeric_bounds_and_cdf_size(by_type):
    numeric = by_type["numeric"]
    assert numeric.cdf_size == 201
    assert numeric.lower_bound < numeric.upper_bound
    assert isinstance(numeric.open_lower_bound, bool) and isinstance(numeric.open_upper_bound, bool)
    assert numeric.zero_point is None


def test_discrete_questions_have_their_own_cdf_size(open_questions):
    sizes = sorted(q.cdf_size for q in open_questions if isinstance(q, DiscreteQuestion))
    assert sizes == [17, 72]
    closed = next(q for q in open_questions if isinstance(q, DiscreteQuestion) and q.cdf_size == 17)
    assert closed.open_lower_bound is False and closed.open_upper_bound is False


def test_date_bounds(by_type):
    date = by_type["date"]
    assert date.cdf_size == 201
    assert date.lower_bound < date.upper_bound
    assert date.open_lower_bound is False and date.open_upper_bound is True


def test_group_posts_are_unpacked_into_sub_questions(questions):
    by_post: dict[int, list] = {}
    for question in questions:
        by_post.setdefault(question.id_of_post, []).append(question)
    groups = {post: qs for post, qs in by_post.items() if len(qs) > 1}
    assert len(groups) == 2
    for subs in groups.values():
        assert len({q.id_of_question for q in subs}) == len(subs)  # distinct question ids, one post id
        assert all(q.group_question_option for q in subs)
        assert all(q.question_ids_of_group == [s.id_of_question for s in subs] for q in subs)
    states = {q.state.value for subs in groups.values() for q in subs}
    assert {"open", "resolved"} <= states  # sub-questions carry their own state


def test_log_scaled_question(posts):
    (question,) = parse_posts([log_scaled_post(posts)])
    assert isinstance(question, NumericQuestion)
    assert question.zero_point == 0.0
    assert (question.lower_bound, question.upper_bound) == (1.0, 1000.0)
    assert question.open_lower_bound is False and question.open_upper_bound is True
