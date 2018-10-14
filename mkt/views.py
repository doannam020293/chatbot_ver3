import csv

from django.http import StreamingHttpResponse
import mysql.connector
from django.http import HttpResponse
import json

cnx = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="hanoi@1993"
)


class Echo:
    """An object that implements just the write method of the file-like
    interface.
    """
    def write(self, value):
        """Write the value by returning it, instead of storing in a buffer."""
        return value

def get_file():


    cursor = cnx.cursor()

    query = ("SELECT *  FROM sys.mkt ")
    cursor.execute(query)
    rows = ([id, sdt, facebook] for (id, sdt, facebook) in cursor)

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