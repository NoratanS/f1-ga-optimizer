"""Tests for schedule formatting utilities."""

from f1_optimizer.cli.schedule import format_schedule, format_schedule_display, get_schedule_summary


class TestFormatSchedule:
    """Tests for format_schedule."""

    def test_race_genes_produce_race_entries_with_correct_track_names(self, track_db, sorted_chromosome):
        # Act
        schedule = format_schedule(sorted_chromosome, track_db)

        # Assert
        races = [e for e in schedule if not e.is_break]
        assert len(races) == len(track_db)
        assert races[0].track_name == track_db[1].track_name

    def test_break_genes_produce_break_entries(self, track_db):
        # Arrange — all breaks chromosome
        chromosome = [100] * 40

        # Act
        schedule = format_schedule(chromosome, track_db)

        # Assert
        assert all(e.is_break for e in schedule)

    def test_week_numbers_are_one_indexed(self, track_db, sorted_chromosome):
        # Act
        schedule = format_schedule(sorted_chromosome, track_db)

        # Assert
        assert schedule[0].week == 1
        assert schedule[-1].week == 40

    def test_total_entries_equal_chromosome_length(self, track_db, sorted_chromosome):
        # Act
        schedule = format_schedule(sorted_chromosome, track_db)

        # Assert
        assert len(schedule) == len(sorted_chromosome)


class TestFormatScheduleDisplay:
    """Tests for format_schedule_display."""

    def test_compact_shows_only_races_without_breaks(self, track_db, sorted_chromosome):
        # Arrange
        schedule = format_schedule(sorted_chromosome, track_db)

        # Act
        display = format_schedule_display(schedule, compact=True)

        # Assert
        assert "Break" not in display
        assert len(display.strip().split("\n")) == len(track_db)

    def test_non_compact_shows_all_forty_weeks(self, track_db, sorted_chromosome):
        # Arrange
        schedule = format_schedule(sorted_chromosome, track_db)

        # Act
        display = format_schedule_display(schedule, compact=False)

        # Assert
        assert len(display.strip().split("\n")) == 40

    def test_compact_races_are_numbered_sequentially_from_one(self, track_db, sorted_chromosome):
        # Arrange
        schedule = format_schedule(sorted_chromosome, track_db)

        # Act
        lines = format_schedule_display(schedule, compact=True).strip().split("\n")

        # Assert
        assert lines[0].startswith("1.")
        assert lines[1].startswith("2.")


class TestGetScheduleSummary:
    """Tests for get_schedule_summary."""

    def test_correct_race_and_break_counts(self, track_db, sorted_chromosome):
        # Arrange
        schedule = format_schedule(sorted_chromosome, track_db)

        # Act
        summary = get_schedule_summary(schedule)

        # Assert
        assert summary["num_races"] == len(track_db)
        assert summary["num_breaks"] == 40 - len(track_db)
        assert summary["total_weeks"] == 40

    def test_first_and_last_race_names(self, track_db, sorted_chromosome):
        # Arrange
        schedule = format_schedule(sorted_chromosome, track_db)

        # Act
        summary = get_schedule_summary(schedule)

        # Assert
        num_races = len(track_db)
        assert summary["first_race"] == track_db[1].track_name
        assert summary["last_race"] == track_db[num_races].track_name
