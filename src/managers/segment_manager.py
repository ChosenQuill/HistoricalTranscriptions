class SegmentManager:
    def __init__(self):
        self.segments = []

    def add_segment(self, points, scan_page_number):
        points = [[float(px), float(py)] for (px, py) in points]
        existing = [s for s in self.segments if s['scan_page'] == scan_page_number]
        seg_id = len(existing) + 1
        self.segments.append({
            'original_points': points,
            'id': seg_id,
            'scan_page': scan_page_number
        })

    def remove_last_segment(self):
        if self.segments:
            self.segments.pop()

    def clear(self):
        self.segments.clear()

    def get_segments(self):
        return self.segments

    def set_segments(self, segments):
        for seg in segments:
            seg['original_points'] = [[float(x), float(y)] for (x, y) in seg['original_points']]
        self.segments = segments

    def update_segment_points(self, seg_id, scan_page_number, new_points):
        new_points = [[float(px), float(py)] for (px, py) in new_points]
        for seg in self.segments:
            if seg['id'] == seg_id and seg['scan_page'] == scan_page_number:
                seg['original_points'] = new_points
                break

    def get_scan_pages(self):
        pages = set(s['scan_page'] for s in self.segments)
        return sorted(list(pages))

    def get_segments_by_scan_page(self, scan_page_number):
        return [s for s in self.segments if s['scan_page'] == scan_page_number]
