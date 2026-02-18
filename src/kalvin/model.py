from dataclasses import dataclass


@dataclass
class KeyValue:
    """A structure with a 64-bit key and list of integer values.

    Attributes:
        key: 64-bit integer key
        value: List of integer values
    """
    key: int          # 64-bit key
    value: list[int]  # list of values

    def matches(self, query: int) -> bool:
        """Check if this KeyValue matches a query via AND operation.

        Args:
            query: The query value to match against

        Returns:
            True if (key & quey) !=0
        """
        return (self.key & query) != 0


def query_by_mask(kv_list: list[KeyValue], query: int) -> list[KeyValue]:
    """Query a list of KeyValues by ANDing keys with a mask.

    Args:
        kv_list: List of KeyValue objects to search
        query: The query value to match
        mask: The 64-bit mask to apply via AND operation

    Returns:
        List of KeyValues where (key & mask) == query
    """
    return [kv for kv in kv_list if kv.matches(query)]
