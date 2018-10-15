import csv

from django.http import StreamingHttpResponse
from django.http import HttpResponse
import json
import _mysql



db=_mysql.connect(host="127.0.0.1",user="useradvdb",
                  passwd="tydHQDP^^###!!18100108",db="adv")





class Echo:
    """An object that implements just the write method of the file-like
    interface.
    """
    def write(self, value):
        """Write the value by returning it, instead of storing in a buffer."""
        return value

def get_file():
    db.query("""SELECT id, sdt, facebook from mkt limit 50""")
    r = db.store_result()
    rows = r.fetch_row(0)

    # cursor.close()
    return list(rows)


def some_streaming_csv_view(request):
    """A view that streams a large CSV file."""
    # Generate a sequence of rows. The range is based on the maximum number of
    # rows that can be handled by a single sheet in most spreadsheet
    # applications.

    rows = get_file()

    pseudo_buffer = Echo()
    writer = csv.writer(pseudo_buffer)
    response = StreamingHttpResponse((writer.writerow(row) for row in rows),
                                     content_type="text/csv")
    response['Content-Disposition'] = 'attachment; filename="hehe.csv"'


    # response = HttpResponse(json.dumps("abc"))

    # cursor.close()
    # cnx.close()

    return response




def test(request):

    return HttpResponse(json.dumps("hello"))