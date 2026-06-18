// src/gfx/bvh.rs
//
// Top-down median-split bounding volume hierarchy over a set of AABBs.
//
// The renderer builds one BVH at GraphicsSystem init time from every cullable
// DrawObject (objects without a finite AABB go into a separate fallback list).
// Each frame the main pass traverses the BVH with the camera's frustum and
// optional distance cutoff, producing a list of DrawObject indices to render.
//
// The BVH is static after construction: it does not refit when leaf AABBs
// change.  Props that move at runtime (held items, animated transforms) must
// opt out of culling by setting a non-finite AABB (UNCULLED_BB in draw_list).
//
// Construction is O(N log N) (sort + recurse); query is O(log N + V) where V
// is the number of visible leaves.  Single-item scenes degenerate to a leaf
// at the root; an empty scene produces no nodes and trivially answers "no
// visible objects".

use crate::gfx::render_types::DrawObject;

use super::frustum::{Frustum, aabb_distance_sq};

#[derive(Copy, Clone, Debug)]
struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
}

impl Aabb {
    fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    fn union(self, other: Aabb) -> Aabb {
        Aabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    fn centroid(self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }
}

#[derive(Debug)]
enum Node {
    Internal {
        bb: Aabb,
        left: u32,
        right: u32,
    },
    Leaf {
        bb: Aabb,
        // Optional per-leaf view-distance cutoff (0 = no cutoff). Stored on
        // the leaf so the traversal can prune without a second array lookup.
        cull_distance: f32,
        index: u32,
    },
}

#[derive(Debug, Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    root: Option<u32>,
}

// One leaf input as supplied to [`Bvh::build`].
#[derive(Copy, Clone, Debug)]
pub struct BvhItem {
    pub bb_min: [f32; 3],
    pub bb_max: [f32; 3],
    pub cull_distance: f32,
    // Caller-side index passed through unchanged to the traversal callback.
    pub index: u32,
}

impl Bvh {
    pub fn build(items: &[BvhItem]) -> Self {
        let mut bvh = Bvh::default();
        if items.is_empty() {
            return bvh;
        }
        let mut working: Vec<BvhItem> = items.to_vec();
        let root = bvh.build_recursive(&mut working);
        bvh.root = Some(root);
        bvh
    }

    // Walk the BVH and call `visit` with every leaf index whose AABB is not
    // fully outside the frustum and whose distance to `cam_pos` is within
    // the leaf's `cull_distance` (0 = always inside).
    pub fn query<F: FnMut(u32)>(&self, frustum: &Frustum, cam_pos: [f32; 3], mut visit: F) {
        let root = match self.root {
            Some(r) => r,
            None => return,
        };
        // Recursive-style traversal using an explicit stack avoids blowing
        // the program stack on degenerate inputs and keeps the hot path
        // amenable to inlining.
        let mut stack: [u32; 64] = [0; 64];
        let mut sp: usize = 0;
        stack[sp] = root;
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let idx = stack[sp];
            match &self.nodes[idx as usize] {
                Node::Internal { bb, left, right } => {
                    if !frustum.intersects_aabb(bb.min, bb.max) {
                        continue;
                    }
                    // Push both children. Order doesn't matter for correctness;
                    // pushing right first means left is visited first (cache
                    // locality if leaves were arranged by build order).
                    if sp + 2 > stack.len() {
                        // Tree depth exceeded the inline stack; fall back to
                        // a heap stack for the rest of this subtree. In
                        // practice 64 entries handles ~2^64 leaves.
                        self.query_heap(idx, frustum, cam_pos, &mut visit);
                        continue;
                    }
                    stack[sp] = *right;
                    sp += 1;
                    stack[sp] = *left;
                    sp += 1;
                }
                Node::Leaf {
                    bb,
                    cull_distance,
                    index,
                } => {
                    if !frustum.intersects_aabb(bb.min, bb.max) {
                        continue;
                    }
                    if *cull_distance > 0.0 {
                        let dsq = aabb_distance_sq(cam_pos, bb.min, bb.max);
                        if dsq > (*cull_distance) * (*cull_distance) {
                            continue;
                        }
                    }
                    visit(*index);
                }
            }
        }
    }

    fn query_heap<F: FnMut(u32)>(
        &self,
        start: u32,
        frustum: &Frustum,
        cam_pos: [f32; 3],
        visit: &mut F,
    ) {
        let mut stack: Vec<u32> = Vec::with_capacity(32);
        stack.push(start);
        while let Some(idx) = stack.pop() {
            match &self.nodes[idx as usize] {
                Node::Internal { bb, left, right } => {
                    if !frustum.intersects_aabb(bb.min, bb.max) {
                        continue;
                    }
                    stack.push(*right);
                    stack.push(*left);
                }
                Node::Leaf {
                    bb,
                    cull_distance,
                    index,
                } => {
                    if !frustum.intersects_aabb(bb.min, bb.max) {
                        continue;
                    }
                    if *cull_distance > 0.0 {
                        let dsq = aabb_distance_sq(cam_pos, bb.min, bb.max);
                        if dsq > (*cull_distance) * (*cull_distance) {
                            continue;
                        }
                    }
                    visit(*index);
                }
            }
        }
    }

    fn build_recursive(&mut self, items: &mut [BvhItem]) -> u32 {
        let bb = items
            .iter()
            .fold(Aabb::empty(), |acc, it| acc.union(item_aabb(it)));

        if items.len() == 1 {
            let it = items[0];
            let node_idx = self.nodes.len() as u32;
            self.nodes.push(Node::Leaf {
                bb,
                cull_distance: it.cull_distance,
                index: it.index,
            });
            return node_idx;
        }

        let extent = [
            bb.max[0] - bb.min[0],
            bb.max[1] - bb.min[1],
            bb.max[2] - bb.min[2],
        ];
        let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
            0
        } else if extent[1] >= extent[2] {
            1
        } else {
            2
        };

        items.sort_by(|a, b| {
            let ca = item_aabb(a).centroid()[axis];
            let cb = item_aabb(b).centroid()[axis];
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = items.len() / 2;
        // Reserve this internal slot first so the indices reflect a stable
        // pre-order layout (root, left subtree, right subtree).
        let self_idx = self.nodes.len() as u32;
        self.nodes.push(Node::Internal {
            bb,
            left: u32::MAX,
            right: u32::MAX,
        });

        let (left_items, right_items) = items.split_at_mut(mid);
        let left = self.build_recursive(left_items);
        let right = self.build_recursive(right_items);

        match &mut self.nodes[self_idx as usize] {
            Node::Internal {
                left: l, right: r, ..
            } => {
                *l = left;
                *r = right;
            }
            Node::Leaf { .. } => unreachable!("self_idx must point at an Internal node"),
        }
        self_idx
    }
}

fn item_aabb(it: &BvhItem) -> Aabb {
    Aabb {
        min: it.bb_min,
        max: it.bb_max,
    }
}

// Partition the draw list into cullable leaves (suitable for BVH insertion)
// and an always-drawn fallback list. Objects that opt out of culling (skybox,
// rooms, held props) keep their original draw order via the returned index
// list; the BVH owns everything else.
pub fn partition_draw_objects(draw_objects: &[DrawObject]) -> (Bvh, Vec<u32>) {
    let mut items: Vec<BvhItem> = Vec::new();
    let mut always_draw: Vec<u32> = Vec::new();
    for (i, obj) in draw_objects.iter().enumerate() {
        if obj.cullable() {
            items.push(BvhItem {
                bb_min: obj.bb_min,
                bb_max: obj.bb_max,
                cull_distance: obj.cull_distance,
                index: i as u32,
            });
        } else {
            always_draw.push(i as u32);
        }
    }
    (Bvh::build(&items), always_draw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident_vp() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn item(idx: u32, min: [f32; 3], max: [f32; 3]) -> BvhItem {
        BvhItem {
            bb_min: min,
            bb_max: max,
            cull_distance: 0.0,
            index: idx,
        }
    }

    fn collect(bvh: &Bvh, frustum: &Frustum) -> Vec<u32> {
        let mut out = Vec::new();
        bvh.query(frustum, [0.0, 0.0, 0.0], |i| out.push(i));
        out.sort();
        out
    }

    #[test]
    fn empty_input_yields_empty_bvh() {
        let bvh = Bvh::build(&[]);
        assert!(bvh.root.is_none());
        assert_eq!(bvh.nodes.len(), 0);
        let f = Frustum::from_view_projection(ident_vp());
        let mut hits = 0;
        bvh.query(&f, [0.0, 0.0, 0.0], |_| hits += 1);
        assert_eq!(hits, 0);
    }

    #[test]
    fn single_item_creates_one_leaf() {
        let bvh = Bvh::build(&[item(7, [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1])]);
        assert!(bvh.root.is_some());
        assert_eq!(bvh.nodes.len(), 1);
        let f = Frustum::from_view_projection(ident_vp());
        assert_eq!(collect(&bvh, &f), vec![7]);
    }

    #[test]
    fn visible_and_invisible_items_separate() {
        let bvh = Bvh::build(&[
            item(0, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
            item(1, [5.0, -0.5, -0.5], [6.0, 0.5, 0.5]),
            item(2, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
        ]);
        let f = Frustum::from_view_projection(ident_vp());
        assert_eq!(collect(&bvh, &f), vec![0, 2]);
    }

    #[test]
    fn distance_cutoff_prunes_far_leaf() {
        let mut a = item(0, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
        a.cull_distance = 1.0;
        let mut b = item(1, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
        b.cull_distance = 0.0;
        let bvh = Bvh::build(&[a, b]);
        let f = Frustum::from_view_projection(ident_vp());
        let mut out = Vec::new();
        // Camera 10 units away: `a` should drop, `b` stays.
        bvh.query(&f, [10.0, 0.0, 0.0], |i| out.push(i));
        out.sort();
        assert_eq!(out, vec![1]);
    }

    #[test]
    fn many_items_all_visible_inside_frustum() {
        let mut items = Vec::new();
        for i in 0..64 {
            let x = (i as f32) * 0.001 - 0.5;
            items.push(item(i, [x, -0.5, -0.5], [x + 0.05, 0.5, 0.5]));
        }
        let bvh = Bvh::build(&items);
        let f = Frustum::from_view_projection(ident_vp());
        let got = collect(&bvh, &f);
        assert_eq!(got.len(), 64);
    }

    #[test]
    fn frustum_rejecting_root_emits_nothing() {
        let bvh = Bvh::build(&[
            item(0, [10.0, -0.5, -0.5], [11.0, 0.5, 0.5]),
            item(1, [10.0, -0.5, -0.5], [11.0, 0.5, 0.5]),
        ]);
        let f = Frustum::from_view_projection(ident_vp());
        assert!(collect(&bvh, &f).is_empty());
    }
}
