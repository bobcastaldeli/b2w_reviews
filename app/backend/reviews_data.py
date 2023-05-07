from pydantic import BaseModel


class ReviewData(BaseModel):
    """This class defines the data that is expected in the request body when
    calling the /predict endpoint."""

    review_text: str
